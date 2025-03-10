import logging
import math

import einops
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former import MaskFormer
from .cl_plugin import build_cl_plugin
from .tracker import build_tracker

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class CTVISTARModel(MaskFormer):
    @configurable
    def __init__(
            self,
            num_frames,
            num_topk,
            num_clip_frames,
            to_cpu_frames,
            test_interpolate_chunk_size,
            test_instance_chunk_size,
            cl_plugin,
            tracker,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.cl_plugin = cl_plugin
        self.tracker = tracker

        self.num_frames = num_frames
        self.num_topk = num_topk
        self.num_clip_frames = num_clip_frames
        self.to_cpu_frames = to_cpu_frames
        self.test_interpolate_chunk_size = test_interpolate_chunk_size
        self.test_instance_chunk_size = test_instance_chunk_size

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.sem_seg_head.pixel_decoder.parameters():
            param.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        rets = MaskFormer.from_config(cfg)  # noqa

        cl_plugin = build_cl_plugin(cfg)  # train
        tracker = build_tracker(cfg)  # inference

        num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        num_topk = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        num_clip_frames = cfg.TEST.NUM_CLIP_FRAMES
        to_cpu_frames = cfg.TEST.TO_CPU_FRAMES
        test_interpolate_chunk_size = cfg.TEST.TEST_INTERPOLATE_CHUNK_SIZE
        test_instance_chunk_size = cfg.TEST.TEST_INSTANCE_CHUNK_SIZE

        rets.update(
            num_frames=num_frames,
            num_topk=num_topk,
            num_clip_frames=num_clip_frames,
            to_cpu_frames=to_cpu_frames,
            test_interpolate_chunk_size=test_interpolate_chunk_size,
            test_instance_chunk_size=test_instance_chunk_size,
            cl_plugin=cl_plugin,
            tracker=tracker
        )

        return rets

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if self.training:
            return self.train_model(batched_inputs)
        else:
            return self.online_inference_model(batched_inputs)
            return self.inference_model(batched_inputs)

    def pre_process(self, batched_inputs):
        images = []  # noqa
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

    def train_model(self, batched_inputs):
        import time
        start = time.time()
        images = self.pre_process(batched_inputs)  # noqa
        images_tensor = images.tensor
        end = time.time()

        features = self.backbone(images_tensor)
        mask_features, _, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)

        mask_features = einops.rearrange(mask_features, '(b t) c h w -> t b c h w', b=len(batched_inputs))
        multi_scale_features = [einops.rearrange(multi_scale_feature, '(b t) c h w -> t b c h w', b=len(batched_inputs)) for multi_scale_feature in multi_scale_features]
        end1 = time.time()

        outputs = [None]
        for idx in range(len(images.tensor) // len(batched_inputs)):
            frame_mask_features = mask_features[idx]
            frame_multi_scale_features = [multi_scale_feature[idx] for multi_scale_feature in multi_scale_features]
            output = self.sem_seg_head.predictor(frame_multi_scale_features, frame_mask_features, outputs[-1])
            outputs.append(output)
        outputs = outputs[1:]

        end2 = time.time()

        new_outputs = {}
        new_outputs['pred_logits'] = torch.cat([output['pred_logits'] for output in outputs], dim=0)
        new_outputs['pred_masks'] = torch.cat([output['pred_masks'] for output in outputs], dim=0)
        new_outputs['pred_embeds'] = torch.cat([output['pred_embeds'] for output in outputs], dim=0)
        new_outputs['pred_queries'] = torch.cat([output['pred_queries'] for output in outputs], dim=0)
        new_outputs['aux_outputs'] = []
        for i in range(len(outputs[0]['aux_outputs'])):
            aux_output = {}
            aux_output['pred_logits'] = torch.cat([output['aux_outputs'][i]['pred_logits'] for output in outputs], dim=0)
            aux_output['pred_masks'] = torch.cat([output['aux_outputs'][i]['pred_masks'] for output in outputs], dim=0)
            aux_output['pred_embeds'] = torch.cat([output['aux_outputs'][i]['pred_embeds'] for output in outputs], dim=0)
            aux_output['pred_queries'] = torch.cat([output['aux_outputs'][i]['pred_queries'] for output in outputs], dim=0)
            new_outputs['aux_outputs'].append(aux_output)

        outputs = new_outputs

        end3 = time.time()

        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = []
            for video in batched_inputs:
                for frame in video["instances"]:
                    gt_instances.append(frame.to(self.device))
            targets = self.prepare_targets(gt_instances, images)
        else:
            targets = None

        end4 = time.time()

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        end5 = time.time()

        losses.update(self.cl_plugin.train_loss(
            outputs, gt_instances, self.criterion.matcher))

        end6 = time.time()

        # print("=====================================")
        # print(f"Preprocess: {end - start}")
        # print(f"pre model time: {end1 - end}")
        # print(f"model time: {end2 - end1}")
        # print(f"post model time: {end3 - end2}")
        # print(f"prepare target time: {end4 - end3}")
        # print(f"loss time: {end5 - end4}")
        # print(f"cl_plugin time: {end6 - end5}")

        return losses

    def online_inference_model(self, batched_inputs):
        images = self.pre_process(batched_inputs)

        # Avoid Out-of-Memory
        num_frames = len(images)
        to_store = self.device if num_frames <= self.to_cpu_frames else "cpu"

        self.num_clip_frames = 1
        class_embed = self.sem_seg_head.predictor.class_embed

        video_dict = {}
        outputs = [None]
        for idx, image in enumerate(images.tensor):
            features = self.backbone(image[None])
            mask_features, _, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(features)
            output = self.sem_seg_head.predictor(multi_scale_features, mask_features, outputs[-1])
            # video_dict is edited
            self.tracker.online_inference(idx, output, class_embed, video_dict)
            outputs.append(output)

        logits_list = []
        masks_list = []
        for inst_id, m in enumerate(video_dict.keys()):
            score_list_ori = video_dict[m]['scores']
            query_list_ori = video_dict[m]['queries']
            mask_list_ori  = video_dict[m]['masks']

            scores_temporal = [score for score in score_list_ori if score is not None]
            queries_temporal = [query for query in query_list_ori if query is not None]
            logits_i = torch.stack(scores_temporal)
            logits_i = logits_i.mean(0) if self.tracker.temporal_score_type == 'mean' else logits_i.max(0)[0]
            logits_list.append(logits_i)

            masks_list_i = [mask if mask is not None else output['pred_masks'].new_zeros(output['pred_masks'].shape[-2:]) for mask in mask_list_ori]
            masks_list_i = torch.stack(masks_list_i, dim=0)
            masks_list.append(masks_list_i)

        if len(logits_list) > 0:
            pred_cls = torch.stack(logits_list, dim=0)[None, ...]
            pred_masks = torch.stack(masks_list, dim=0)[None, ...]
        else:
            pred_cls = []

        outputs = {
            'pred_logits': pred_cls,
            'pred_masks': pred_masks
        }

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        input_per_image = batched_inputs[0]
        image_tensor_size = images.tensor.shape
        # image size without padding after data augmentation
        image_size = images.image_sizes[0]

        # raw image size before data augmentation
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])

        del outputs, batched_inputs, images

        video_output = self.inference_video(mask_cls_results, mask_pred_results, image_tensor_size, image_size,
                                            height, width, to_store)
        return video_output

    def inference_model(self, batched_inputs):
        images = self.pre_process(batched_inputs)

        # Avoid Out-of-Memory
        num_frames = len(images)
        to_store = self.device if num_frames <= self.to_cpu_frames else "cpu"

        if num_frames <= self.num_clip_frames:  # noqa
            with torch.no_grad():
                features = self.backbone(images.tensor)
                det_outputs = self.sem_seg_head(features)
        else:
            pred_logits, pred_masks, pred_embeds, pred_queries = [], [], [], []
            num_clips = math.ceil(
                num_frames / self.num_clip_frames)  # math.ceil up
            for i in range(num_clips):
                start_idx = i * self.num_clip_frames
                end_idx = (i + 1) * self.num_clip_frames
                clip_images_tensor = images.tensor[start_idx:end_idx, ...]
                with torch.no_grad():
                    clip_features = self.backbone(clip_images_tensor)
                    clip_outputs = self.sem_seg_head(clip_features)

                pred_logits.append(clip_outputs['pred_logits'])
                pred_masks.append(clip_outputs['pred_masks'])
                pred_embeds.append(clip_outputs['pred_embeds'])
                pred_queries.append(clip_outputs['pred_queries'])

            det_outputs = {
                'pred_logits': torch.cat(pred_logits, dim=0),
                'pred_masks': torch.cat(pred_masks, dim=0),
                'pred_embeds': torch.cat(pred_embeds, dim=0),
                'pred_queries': torch.cat(pred_queries, dim=0)
            }

        class_embed = self.sem_seg_head.predictor.class_embed
        outputs = self.tracker.inference(det_outputs, hybrid_embed=class_embed)

        print(torch.allclose(outputs['pred_logits'], self.outputs['pred_logits']))
        print(torch.allclose(outputs['pred_masks'], self.outputs['pred_masks']))

        print(outputs['pred_logits'].shape, outputs['pred_masks'].shape)
        exit(0)

        if len(outputs['pred_logits']) == 0:
            video_output = {
                "image_size":  (images.image_sizes[0], images.image_sizes[1]),
                "pred_scores": [],
                "pred_labels": [],
                "pred_masks":  []
            }
            return video_output

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        input_per_image = batched_inputs[0]
        image_tensor_size = images.tensor.shape
        # image size without padding after data augmentation
        image_size = images.image_sizes[0]

        # raw image size before data augmentation
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])

        del outputs, batched_inputs, images, det_outputs

        video_output = self.inference_video(mask_cls_results, mask_pred_results, image_tensor_size, image_size,
                                            height, width, to_store)

        return video_output

    def inference_video(self, mask_cls_results, mask_pred_results, image_tensor_size, image_size, height, width,
                        to_store):
        mask_cls_result = mask_cls_results[0]
        mask_pred_result = mask_pred_results[0]

        if len(mask_cls_result) > 0:
            scores = F.softmax(mask_cls_result, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
                len(mask_cls_result), 1).flatten(0, 1)  # noqa
            scores_per_video, topk_indices = scores.flatten(
                0, 1).topk(self.num_topk, sorted=True)

            labels_per_video = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes

            mask_pred_result = mask_pred_result[topk_indices]

            num_instance, num_frame = mask_pred_result.shape[:2]

            out_scores = []
            out_labels = []
            out_masks = []

            for k in range(math.ceil(num_instance / self.test_instance_chunk_size)):
                _mask_pred_result = mask_pred_result[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]
                _scores_per_video = scores_per_video[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]
                _labels_per_video = labels_per_video[
                    k * self.test_instance_chunk_size:(k + 1) * self.test_instance_chunk_size, ...]

                masks_list = []  # noqa
                numerator = torch.zeros(
                    _mask_pred_result.shape[0], dtype=torch.float, device=self.device)
                denominator = torch.zeros(
                    _mask_pred_result.shape[0], dtype=torch.float, device=self.device)
                for i in range(math.ceil(num_frame / self.test_interpolate_chunk_size)):
                    temp_pred_mask = _mask_pred_result[:,
                                     i * self.test_interpolate_chunk_size:(i + 1) * self.test_interpolate_chunk_size,
                                     ...]  # noqa
                    temp_pred_mask = retry_if_cuda_oom(F.interpolate)(
                        temp_pred_mask,
                        size=(image_tensor_size[-2], image_tensor_size[-1]),
                        mode="bilinear",
                        align_corners=False)
                    temp_pred_mask = temp_pred_mask[:, :,
                                                    : image_size[0], : image_size[1]]

                    temp_pred_mask = retry_if_cuda_oom(F.interpolate)(temp_pred_mask, size=(height, width),
                                                                      mode="bilinear", align_corners=False)  # noqa
                    masks = (temp_pred_mask > 0.).float()
                    numerator += (temp_pred_mask.sigmoid()
                                  * masks).flatten(1).sum(1)
                    denominator += masks.flatten(1).sum(1)

                    masks_list.append(masks.bool().to(to_store))
                _scores_per_video *= (numerator / (denominator + 1e-6))
                masks = torch.cat(masks_list, dim=1)

                out_scores.extend(_scores_per_video.tolist())
                out_labels.extend(_labels_per_video.tolist())
                out_masks.extend([m for m in masks.cpu()])
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (height, width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }
        return video_output

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            if isinstance(targets_per_image.gt_masks, BitMasks):
                gt_masks = targets_per_image.gt_masks.tensor
            else:
                gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1],
                         : gt_masks.shape[2]] = gt_masks

            # filter empty instances
            gt_instance_ids = targets_per_image.gt_ids
            valid_index = gt_instance_ids != -1

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes[valid_index],
                    "masks": padded_masks[valid_index],
                    "ids": gt_instance_ids[valid_index],
                }
            )
        return new_targets
