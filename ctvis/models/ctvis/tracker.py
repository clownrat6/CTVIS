import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from detectron2.config import configurable
from detectron2.utils.registry import Registry

TRACKER_REGISTRY = Registry("Tracker")
TRACKER_REGISTRY.__doc__ = """Registry for Tracker for Online Video Instance Segmentation Models."""


def build_tracker(cfg):
    """
    Build a tracker for online video instance segmentation models
    """
    name = cfg.MODEL.TRACKER.TRACKER_NAME
    return TRACKER_REGISTRY.get(name)(cfg)


def mask_iou(mask1, mask2):
    mask1 = mask1.char()
    mask2 = mask2.char()

    intersection = (mask1[:, :, :] * mask2[:, :, :]).sum(-1).sum(-1)
    union = (mask1[:, :, :] + mask2[:, :, :] -
             mask1[:, :, :] * mask2[:, :, :]).sum(-1).sum(-1)

    return (intersection + 1e-6) / (union + 1e-6)


def mask_nms(seg_masks, scores, nms_thr=0.5):
    n_samples = len(scores)
    if n_samples == 0:
        return []
    keep = [True for i in range(n_samples)]
    seg_masks = seg_masks.sigmoid() > 0.5

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        # label_i = cate_labels[i]
        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]

            iou = mask_iou(mask_i, mask_j)[0]
            if iou > nms_thr:
                keep[j] = False
    return keep


class Tracklet(object):
    """
    用于定义一个track 片段， 表示一个 小的跟踪片段（轨迹），是组成一个memory bank的基本单元
    """

    def __init__(self, instance_id, maximum_cache=10, linear=None):
        self.instance_id = instance_id
        self.logits = []
        self.masks = []
        self.reid_embeds = []
        self.long_scores = []
        self.frame_ids = []
        self.last_reid_embed = torch.zeros((256,), device='cuda')
        self.similarity_guided_reid_embed = torch.zeros((256,), device='cuda')
        self.fusion_reid_embed = torch.zeros((256,), device='cuda')
        self.exist_frames = 0
        self.maximum_cache = maximum_cache
        self.momentum = 0.75  # 这个效果是最好的
        self.linear = linear  # todo: 使用线性层代替 动量更新

    def update(self, score, logit, mask, reid_embed, frame_id):
        # 用于更新 track时使用
        self.long_scores.append(score)
        self.logits.append(logit)
        self.masks.append(mask)
        self.reid_embeds.append(reid_embed)
        self.frame_ids.append(frame_id)

        if self.exist_frames == 0:
            # 第一次出现的话直接使用使用
            # 加不加差不多 (+ 0.02)
            self.last_reid_embed = reid_embed
            self.similarity_guided_reid_embed = reid_embed
            self.fusion_reid_embed = reid_embed
        else:
            self.last_reid_embed = (1 - self.momentum) * self.last_reid_embed + self.momentum * reid_embed

            # Similarity-Guided Feature Fusion
            # https://arxiv.org/abs/2203.14208v1
            all_reid_embed = torch.stack(self.reid_embeds[:-1], dim=0)
            similarity = torch.sum(torch.einsum("bc,c->b",
                                                F.normalize(all_reid_embed, dim=-1),
                                                F.normalize(reid_embed, dim=-1))) / (len(self.reid_embeds) - 1)
            beta = max(0, similarity)
            self.similarity_guided_reid_embed = (1 - beta) * self.similarity_guided_reid_embed + beta * reid_embed
            # 用 可学习的优化来进行更新
            if self.linear:
                self.fusion_reid_embed = self.linear(self.fusion_reid_embed + reid_embed)

        self.exist_frames += 1

        if len(self.long_scores) > self.maximum_cache:
            self.long_scores.pop(0)
            self.reid_embeds.pop(0)


class MemoryBank:
    """
    众所周知，就是一个memory bank，主要用来存取 tracklet，与 CL 中的 Memory Bank 有一定的区别
    """

    def __init__(self,
                 embed_type='temporally_weighted_softmax',
                 num_dead_frames=10,
                 maximum_cache=10):
        self.tracklets = dict()
        self.num_tracklets = 0

        self.embed_type = embed_type  # last | temporally_weighted_softmax | momentum | similarity_guided
        self.num_dead_frames = num_dead_frames
        self.maximum_cache = maximum_cache

    def add(self, instance_id):
        self.tracklets[instance_id] = Tracklet(instance_id, self.maximum_cache)
        self.num_tracklets += 1
        # self.num_tracklets += 0  # bug fixed

    def update(self, instance_id, score, logit, mask, reid_embed, frame_id):
        self[instance_id].update(score, logit, mask, reid_embed, frame_id)

    def __getitem__(self, instance_id):
        return self.tracklets[instance_id]

    def __len__(self):
        return self.num_tracklets

    @property
    def exist_ids(self):
        return self.tracklets.keys()

    def clean_dead_tracklets(self, cur_frame_id):
        # 删除已经死亡的 tracklets
        dead_ids = []
        for instance_id, tracklet in self.tracklets.items():
            if cur_frame_id - tracklet.frame_ids[-1] > self.num_dead_frames:
                dead_ids.append(instance_id)
        for dead_id in dead_ids:
            del self.tracklets[dead_id]
            self.num_tracklets -= 1

    def exist_reid_embeds(self, frame_id):
        memory_bank_embeds = []
        memory_bank_ids = []
        memory_bank_exist_frames = []
        for instance_id, tracklet in self.tracklets.items():
            if self.embed_type == 'temporally_weighted_softmax':
                score_weights = torch.stack(tracklet.long_scores)
                length = score_weights.shape[0]
                temporal_weights = torch.range(0., 1, 1 / length)[1:].to(score_weights)
                weights = score_weights + temporal_weights
                weighted_sum_embed = (torch.stack(tracklet.reid_embeds) * weights.unsqueeze(1)).sum(0) / weights.sum()
                memory_bank_embeds.append(weighted_sum_embed)
            elif self.embed_type == 'last':
                memory_bank_embeds.append(tracklet.reid_embeds[-1])
            elif self.embed_type == 'momentum':  # 动量更新
                memory_bank_embeds.append(tracklet.last_reid_embed)
            elif self.embed_type == 'similarity_guided':
                memory_bank_embeds.append(tracklet.similarity_guided_reid_embed)
            else:
                raise NotImplementedError

            memory_bank_ids.append(instance_id)
            # num_frame_disappear = float(frame_id - tracklet.frame_ids[-1])
            # memory_bank_exist_frames.append(max(tracklet.exist_frames / num_frame_disappear, 0.8))
            memory_bank_exist_frames.append(tracklet.exist_frames)

        memory_bank_embeds = torch.stack(memory_bank_embeds, dim=0)
        memory_bank_ids = memory_bank_embeds.new_tensor(memory_bank_ids).to(dtype=torch.long)
        # memory_bank_exist_frames = memory_bank_embeds.new_tensor(memory_bank_exist_frames).to(dtype=torch.float32)
        memory_bank_exist_frames = memory_bank_embeds.new_tensor(memory_bank_exist_frames).to(dtype=torch.long)

        return memory_bank_ids, memory_bank_embeds, memory_bank_exist_frames


@TRACKER_REGISTRY.register()
class SimpleTracker(nn.Module):
    """
    Simple Tracker for Online Video Instance Segmentation.
    Follow IDOL.
    """

    @configurable
    def __init__(self,
                 *,
                 num_classes,
                 match_metric,
                 frame_weight,
                 match_score_thr,
                 temporal_score_type,
                 match_type,
                 inference_select_thr,
                 init_score_thr,
                 mask_nms_thr,
                 num_dead_frames,
                 embed_type,
                 maximum_cache,
                 noise_frame_num,
                 noise_frame_ratio,
                 suppress_frame_num,
                 none_frame_num):
        super().__init__()
        self.num_classes = num_classes
        self.match_metric = match_metric
        self.match_score_thr = match_score_thr
        self.temporal_score_type = temporal_score_type
        self.temporal_score_type = temporal_score_type
        assert self.temporal_score_type in ['mean', 'max', 'hybrid']
        self.match_type = match_type  # greedy hungarian

        self.inference_select_thr = inference_select_thr
        self.init_score_thr = init_score_thr
        self.mask_nms_thr = mask_nms_thr
        self.frame_weight = frame_weight

        self.num_dead_frames = num_dead_frames
        self.embed_type = embed_type
        self.maximum_cache = maximum_cache

        self.noise_frame_num = noise_frame_num
        self.noise_frame_ratio = noise_frame_ratio
        self.suppress_frame_num = suppress_frame_num
        self.none_frame_num = none_frame_num

    @classmethod
    def from_config(cls, cfg):
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES  # noqa

        match_metric = cfg.MODEL.TRACKER.MATCH_METRIC
        frame_weight = cfg.MODEL.TRACKER.FRAME_WEIGHT
        match_score_thr = cfg.MODEL.TRACKER.MATCH_SCORE_THR
        match_type = cfg.MODEL.TRACKER.MATCH_TYPE
        inference_select_thr = cfg.MODEL.TRACKER.INFERENCE_SELECT_THR
        init_score_thr = cfg.MODEL.TRACKER.INIT_SCORE_THR
        mask_nms_thr = cfg.MODEL.TRACKER.MASK_NMS_THR
        temporal_score_type = cfg.MODEL.TRACKER.TEMPORAL_SCORE_TYPE

        num_dead_frames = cfg.MODEL.TRACKER.MEMORY_BANK.NUM_DEAD_FRAMES
        embed_type = cfg.MODEL.TRACKER.MEMORY_BANK.EMBED_TYPE
        maximum_cache = cfg.MODEL.TRACKER.MEMORY_BANK.maximum_cache

        noise_frame_num    = 8
        noise_frame_ratio  = 0.4
        suppress_frame_num = 3
        none_frame_num     = 2

        ret = {
            "num_classes": num_classes,
            "match_metric": match_metric,
            "frame_weight": frame_weight,
            "match_score_thr": match_score_thr,
            "temporal_score_type": temporal_score_type,
            "match_type": match_type,
            "inference_select_thr": inference_select_thr,
            "init_score_thr": init_score_thr,
            "mask_nms_thr": mask_nms_thr,
            # memory bank & tracklet
            "num_dead_frames": num_dead_frames,
            "embed_type": embed_type,
            "maximum_cache": maximum_cache,
            "noise_frame_num": noise_frame_num,
            "noise_frame_ratio": noise_frame_ratio,
            "suppress_frame_num": suppress_frame_num,
            "none_frame_num": none_frame_num
        }
        return ret

    @property
    def device(self):
        return torch.device('cuda')

    @property
    def empty(self):
        return self.num_tracklets == 0

    def reset(self):
        self.num_tracklets = 0  # noqa
        self.memory_bank = MemoryBank(self.embed_type,  # noqa
                                      self.num_dead_frames,
                                      self.maximum_cache)

    def update(self, ids, pred_scores, pred_logits, pred_masks, pred_reid_embeds, frame_id):
        assert ids.shape[0] == pred_logits.shape[0], 'Shape must match.'  # noqa

        num_instances = ids.shape[0]

        for instance_index in range(num_instances):

            instance_id = int(ids[instance_index].item())
            instance_score = pred_scores[instance_index]
            instance_logit = pred_logits[instance_index]
            instance_mask = pred_masks[instance_index]
            instance_reid_embed = pred_reid_embeds[instance_index]

            if instance_id in self.memory_bank.exist_ids:
                self.memory_bank.update(instance_id, instance_score, instance_logit,
                                        instance_mask, instance_reid_embed, frame_id)
            else:
                self.memory_bank.add(instance_id)
                self.memory_bank.update(instance_id, instance_score, instance_logit,
                                        instance_mask, instance_reid_embed, frame_id)

    def online_inference(self, frame_id, det_outputs, hybrid_embed, video_dict):
        if frame_id == 0:
            self.reset()

        pred_logits  = det_outputs['pred_logits'][0]
        pred_masks   = det_outputs['pred_masks'][0]
        pred_embeds  = det_outputs['pred_embeds'][0]
        pred_queries = det_outputs['pred_queries'][0]

        scores = F.softmax(pred_logits, dim=-1)[:, :-1]
        max_scores, max_indices = torch.max(scores, dim=1)
        _, sorted_indices = torch.sort(max_scores, dim=0, descending=True)

        pred_scores  = max_scores[sorted_indices]
        pred_logits  = pred_logits[sorted_indices]
        pred_masks   = pred_masks[sorted_indices]
        pred_embeds  = pred_embeds[sorted_indices]
        pred_queries = pred_queries[sorted_indices]

        valid_indices = pred_scores > self.inference_select_thr
        valid_indices[0] = 1 if valid_indices.sum() == 0 else valid_indices[0]

        pred_scores  = pred_scores[valid_indices]
        pred_logits  = pred_logits[valid_indices]
        pred_masks   = pred_masks[valid_indices]
        pred_embeds  = pred_embeds[valid_indices]
        pred_queries = pred_queries[valid_indices]

        # NMS: can bring a slight improvement
        valid_nms_indices = mask_nms(pred_masks[:, None, ...], pred_scores, nms_thr=self.mask_nms_thr)

        pred_scores  = pred_scores[valid_nms_indices]
        pred_logits  = pred_logits[valid_nms_indices]
        pred_masks   = pred_masks[valid_nms_indices]
        pred_embeds  = pred_embeds[valid_nms_indices]
        pred_queries = pred_queries[valid_nms_indices]

        ids, pred_logits, pred_masks, pred_queries = \
            self.track(pred_scores, pred_logits, pred_masks, pred_embeds, pred_queries, frame_id)

        for index in range(ids.shape[0]):
            instance_id = ids[index]
            pred_logit  = pred_logits[index]
            pred_mask   = pred_masks[index]
            pred_query  = pred_queries[index]

            if instance_id.item() in video_dict.keys():
                video_dict[instance_id.item()]['masks'].append(pred_mask)
                video_dict[instance_id.item()]['scores'].append(pred_logit)
                video_dict[instance_id.item()]['queries'].append(pred_query)
            else:
                video_dict[instance_id.item()] = {
                    'masks': [None for _ in range(frame_id)],
                    'scores': [None for _ in range(frame_id)],
                    'queries': [None for _ in range(frame_id)]}
                video_dict[instance_id.item()]['masks'].append(pred_mask)
                video_dict[instance_id.item()]['scores'].append(pred_logit)
                video_dict[instance_id.item()]['queries'].append(pred_query)

        for k, v in video_dict.items():
            if len(v['masks']) < frame_id + 1:
                v['masks'].append(None)
                v['scores'].append(None)
                v['queries'].append(None)

        # filter sequences that are too short in video_dict (noise)，
        # the rule is: if the first two frames are None and valid is less than 3
        # stolen from IDOL
        # noise_frame_num = math.floor(num_frames * self.noise_frame_ratio)
        if frame_id > self.noise_frame_num:
            del_list = []
            for k, v in video_dict.items():
                valid = sum([1 if _ is not None else 0 for _ in v['masks']])
                none_frame = 0
                for m in v['masks'][::-1]:
                    if m is None:
                        none_frame += 1
                    else:
                        break
                # describe the code: if the last 3 frames are None and valid is less than 3
                if none_frame >= self.none_frame_num and valid < self.suppress_frame_num:
                    del_list.append(k)
            for del_k in del_list:
                video_dict.pop(del_k)
                # self.memory_bank.delete_tracklet(del_k)  uncomment will drop 0.24 AP

    def inference(self, det_outputs, hybrid_embed):
        num_frames, num_queries = det_outputs['pred_logits'].shape[:2]

        video_dict = dict()
        for frame_id in range(num_frames):
            if frame_id == 0:
                self.reset()

            pred_logits = det_outputs['pred_logits'][frame_id]
            pred_masks = det_outputs['pred_masks'][frame_id]
            pred_embeds = det_outputs['pred_embeds'][frame_id]
            pred_queries = det_outputs['pred_queries'][frame_id]

            scores = F.softmax(pred_logits, dim=-1)[:, :-1]
            max_scores, max_indices = torch.max(scores, dim=1)
            _, sorted_indices = torch.sort(max_scores, dim=0, descending=True)

            pred_scores = max_scores[sorted_indices]
            pred_logits = pred_logits[sorted_indices]
            pred_masks = pred_masks[sorted_indices]
            pred_embeds = pred_embeds[sorted_indices]
            pred_queries = pred_queries[sorted_indices]

            valid_indices = pred_scores > self.inference_select_thr
            if valid_indices.sum() == 0:
                valid_indices[0] = 1
            pred_scores = pred_scores[valid_indices]
            pred_logits = pred_logits[valid_indices]
            pred_masks = pred_masks[valid_indices]
            pred_embeds = pred_embeds[valid_indices]
            pred_queries = pred_queries[valid_indices]

            # NMS: can bring a slight improvement
            valid_nms_indices = mask_nms(pred_masks[:, None, ...], pred_scores, nms_thr=self.mask_nms_thr)
            pred_scores = pred_scores[valid_nms_indices]
            pred_logits = pred_logits[valid_nms_indices]
            pred_masks = pred_masks[valid_nms_indices]
            pred_embeds = pred_embeds[valid_nms_indices]
            pred_queries = pred_queries[valid_nms_indices]

            ids, pred_logits, pred_masks, pred_queries = \
                self.track(pred_scores, pred_logits, pred_masks, pred_embeds, pred_queries, frame_id)

            for index in range(ids.shape[0]):
                instance_id = ids[index]
                pred_logit = pred_logits[index]
                pred_mask = pred_masks[index]
                pred_query = pred_queries[index]

                if instance_id.item() in video_dict.keys():
                    video_dict[instance_id.item()]['masks'].append(pred_mask)
                    video_dict[instance_id.item()]['scores'].append(pred_logit)
                    video_dict[instance_id.item()]['queries'].append(pred_query)
                else:
                    # NOTE: if the instance_id is first appeared in the current frame, the previous frames should be filled with None
                    video_dict[instance_id.item()] = {
                        'masks':   [None for _ in range(frame_id)] + [pred_mask],
                        'scores':  [None for _ in range(frame_id)] + [pred_logit],
                        'queries': [None for _ in range(frame_id)] + [pred_query]
                    }

            for k, v in video_dict.items():
                if len(v['masks']) < frame_id + 1:
                    v['masks'].append(None)
                    v['scores'].append(None)
                    v['queries'].append(None)

            # filter sequences that are too short in video_dict (noise)，
            # the rule is: if the first two frames are None and valid is less than 3
            # stolen from IDOL
            # noise_frame_num = math.floor(num_frames * self.noise_frame_ratio)
            if frame_id > self.noise_frame_num:
                del_list = []
                for k, v in video_dict.items():
                    valid = sum([1 if _ is not None else 0 for _ in v['masks']])
                    none_frame = 0
                    for m in v['masks'][::-1]:
                        if m is None:
                            none_frame += 1
                        else:
                            break
                    if none_frame >= self.none_frame_num and valid < self.suppress_frame_num:
                        del_list.append(k)
                for del_k in del_list:
                    video_dict.pop(del_k)
                    # self.memory_bank.delete_tracklet(del_k)  uncomment will drop 0.24 AP

        logits_list = []
        masks_list = []
        mask_h, mask_w = det_outputs['pred_masks'].shape[-2:]
        for inst_id, m in enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            query_list_ori = video_dict[m]['queries']
            scores_temporal = []
            queries_temporal = []
            for t, k in zip(query_list_ori, score_list_ori):
                if k is not None:
                    scores_temporal.append(k)
                    queries_temporal.append(t)
            logits_i = torch.stack(scores_temporal)
            if self.temporal_score_type == 'mean':
                logits_i = logits_i.mean(0)
            elif self.temporal_score_type == 'max':
                logits_i = logits_i.max(0)[0]
            elif self.temporal_score_type == 'hybrid':
                raise NotImplementedError
            logits_list.append(logits_i)

            masks_list_i = []
            for n in range(num_frames):
                mask_i = video_dict[m]['masks'][n]
                if mask_i is None:
                    zero_mask = det_outputs['pred_masks'].new_zeros(mask_h, mask_w)
                    masks_list_i.append(zero_mask)
                else:
                    masks_list_i.append(mask_i)
            masks_list_i = torch.stack(masks_list_i, dim=0)
            masks_list.append(masks_list_i)
        if len(logits_list) > 0:
            pred_cls = torch.stack(logits_list, dim=0)[None, ...]
            pred_masks = torch.stack(masks_list, dim=0)[None, ...]
        else:
            pred_cls = []

        return {
            'pred_logits': pred_cls,
            'pred_masks': pred_masks
        }

    def track(self, pred_scores, pred_logits, pred_masks, pred_embeds, pred_queries, frame_id):
        ids = pred_logits.new_full((pred_logits.shape[0],), -1, dtype=torch.long)

        if self.empty:
            valid_init_indices = pred_scores > self.init_score_thr
            num_new_tracklets = valid_init_indices.sum()
            ids[valid_init_indices] = torch.arange(self.num_tracklets, self.num_tracklets + num_new_tracklets,
                                                   dtype=torch.long).to(self.device)
            self.num_tracklets += num_new_tracklets

        else:
            num_instances = pred_logits.shape[0]
            exist_tracklet_ids, exist_reid_embeds, exist_frames = self.memory_bank.exist_reid_embeds(frame_id)

            if self.match_metric == 'bisoftmax':
                # d2t: 45.6
                # t2d: 45.7
                # bio: (t2d + d2t) / 2 : 48.3  good
                similarity = torch.mm(pred_embeds, exist_reid_embeds.t())
                d2t_scores = similarity.softmax(dim=1)
                t2d_scores = similarity.softmax(dim=0)
                match_scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'cosine':
                key = F.normalize(pred_embeds, p=2, dim=1)
                query = F.normalize(exist_reid_embeds, p=2, dim=1)
                match_scores = torch.mm(key, query.t())
            else:
                raise NotImplementedError

            if self.match_type == 'greedy':
                for idx in range(num_instances):
                    if self.frame_weight:
                        valid_indices = match_scores[idx, :] > self.match_score_thr
                        if (match_scores[idx, valid_indices] > self.match_score_thr).sum() > 1:
                            wighted_scores = match_scores.clone()
                            frame_weight = exist_frames[valid_indices].to(wighted_scores)
                            wighted_scores[idx, valid_indices] = wighted_scores[idx, valid_indices] * frame_weight
                            wighted_scores[idx, ~valid_indices] = wighted_scores[
                                                                      idx, ~valid_indices] * frame_weight.mean()
                            match_score, max_indices = torch.max(wighted_scores[idx, :], dim=0)
                        else:
                            match_score, max_indices = torch.max(match_scores[idx, :], dim=0)
                    else:
                        match_score, max_indices = torch.max(match_scores[idx, :], dim=0)

                    match_tracklet_id = exist_tracklet_ids[max_indices]
                    assert match_tracklet_id > -1
                    if match_score > self.match_score_thr:
                        ids[idx] = match_tracklet_id
                        match_scores[:idx, max_indices] = 0
                        match_scores[idx + 1:, max_indices] = 0
            elif self.match_type == 'hungarian':
                # drop 3 AP
                match_cost = - match_scores.cpu()
                indices = linear_sum_assignment(match_cost)

                for i, (instance_id, _id) in enumerate(zip(*indices)):
                    if match_scores[instance_id, _id] < self.match_score_thr:
                        indices[1][i] = -1

                ids[indices[0]] = ids.new_tensor(exist_tracklet_ids[indices[1]])
                ids[indices[0][indices[1] == -1]] = -1
            else:
                raise NotImplementedError

            new_instance_indices = (ids == -1) & (pred_scores > self.init_score_thr)
            num_new_tracklets = new_instance_indices.sum().item()
            ids[new_instance_indices] = torch.arange(self.num_tracklets,
                                                     self.num_tracklets + num_new_tracklets,
                                                     dtype=torch.long).to(self.device)
            self.num_tracklets += num_new_tracklets

        valid_inds = ids > -1
        ids = ids[valid_inds]
        pred_scores = pred_scores[valid_inds]
        pred_logits = pred_logits[valid_inds]
        pred_masks = pred_masks[valid_inds]
        pred_embeds = pred_embeds[valid_inds]
        pred_queries = pred_queries[valid_inds]

        self.update(
            ids=ids,
            pred_scores=pred_scores,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            pred_reid_embeds=pred_embeds,
            frame_id=frame_id)
        self.memory_bank.clean_dead_tracklets(frame_id)

        return ids, pred_logits, pred_masks, pred_queries
