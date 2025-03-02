import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch.nn import functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.structures import BitMasks
from detectron2.layers import Conv2d

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder  import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import _get_activation_fn, SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from .cl_mask2former_transformer_decoder import CLMultiScaleMaskedTransformerDecoder


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    # index = y_soft.max(dim, keepdim=True)[1]
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)

    return y_hard - y_soft.detach() + y_soft


class LayerRouter(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

        self.router = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
        )

    def forward(self, x, last_x):
        router_logits = torch.cat([x, last_x, x - last_x], dim=-1)
        router_logits = self.router(router_logits)
        router_logits = router_logits.mean(dim=0)
        # soft argmax
        router_weights = hard_softmax(router_logits, dim=-1)

        return router_weights, router_logits


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, tgt_pos=None, **kwargs):
        q = k = tgt if tgt_pos is None else tgt + tgt_pos
        tgt2  = self.self_attn(q, k, value=tgt, need_weights=False, **kwargs)[0]
        tgt   = tgt + self.dropout(tgt2)
        tgt   = self.norm(tgt)
        return tgt


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, tgt_pos=None, memory_pos=None, **kwargs):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, need_weights=False, **kwargs)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


@TRANSFORMER_DECODER_REGISTRY.register()
class CLMultiScaleMaskedTransformerDecoderTAR(CLMultiScaleMaskedTransformerDecoder):
    
    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transformer_self_attention_layers = nn.ModuleList([SelfAttentionLayer(d_model=self.hidden_dim, nhead=self.num_heads) for _ in range(self.num_layers)])
        self.transformer_cross_attention_layers = nn.ModuleList([CrossAttentionLayer(d_model=self.hidden_dim, nhead=self.num_heads) for _ in range(self.num_layers)])
        self.transformer_ffn_layers = nn.ModuleList([FFNLayer(d_model=self.hidden_dim, dim_feedforward=self.dim_feedforward) for _ in range(self.num_layers)])

        self.split_stage = 1

        self.layer_router = LayerRouter(self.hidden_dim)
        self.sampler = CrossAttentionLayer(self.hidden_dim, self.num_heads)

    def forward_decoder_layer(self, idx, tgt, tgt_pos, memory, memory_pos, attn_mask):
        tgt = self.transformer_cross_attention_layers[idx](
            tgt, memory,
            tgt_pos=tgt_pos, memory_pos=memory_pos,
            attn_mask=attn_mask, key_padding_mask=None,
        )

        tgt = self.transformer_self_attention_layers[idx](
            tgt,
            tgt_pos=tgt_pos,
            attn_mask=None, key_padding_mask=None,
        )

        # FFN
        tgt = self.transformer_ffn_layers[idx](tgt)

        return tgt

    def forward(self, x, mask_features, old_outputs=None):
        # prepare time: 0.00169s
        level_embed_weight = self.level_embed.weight[:, None, :, None]

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + level_embed_weight[i])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        tgt     = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, outputs_query, outputs_embed = \
            self.forward_prediction_heads(tgt, mask_features, attn_mask_target_size=size_list[0])
        predictions_class = [outputs_class]
        predictions_mask  = [outputs_mask]
        predictions_query = [outputs_query]
        predictions_embed = [outputs_embed]

        for i in range(self.split_stage):
            level_index = i % self.num_feature_levels
            tgt = self.forward_decoder_layer(i, tgt, tgt_pos, src[level_index], pos[level_index], attn_mask)

            outputs_class, outputs_mask, attn_mask, outputs_query, outputs_embed = \
                self.forward_prediction_heads(tgt, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(outputs_query)
            predictions_embed.append(outputs_embed)

        old_tgt = old_outputs['pred_queries'].transpose(0, 1) if old_outputs is not None else tgt
        tgt = self.sampler(old_tgt, tgt)

        outputs_class, outputs_mask, attn_mask, outputs_query, outputs_embed = \
                self.forward_prediction_heads(tgt, mask_features, attn_mask_target_size=size_list[self.split_stage % self.num_feature_levels])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_query.append(outputs_query)
        predictions_embed.append(outputs_embed)

        router_logits_list = []

        for i in range(self.split_stage, self.num_layers):
            level_index = i % self.num_feature_levels
            router_weights, router_logits = self.layer_router(tgt, old_tgt)
            router_logits_list.append(router_logits)
            residual = tgt

            if self.training:
                tgt = self.forward_decoder_layer(i, tgt, tgt_pos, src[level_index], pos[level_index], attn_mask)
                tgt = router_weights[:, 0, None] * residual + router_weights[:, 1, None] * tgt

                outputs_class, outputs_mask, attn_mask, outputs_query, outputs_embed = \
                        self.forward_prediction_heads(tgt, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)
                predictions_query.append(outputs_query)
                predictions_embed.append(outputs_embed)
            else:
                if router_weights[0, 1] == 1:
                    tgt = self.forward_decoder_layer(i, tgt, tgt_pos, src[level_index], pos[level_index], attn_mask)
                    outputs_class, outputs_mask, attn_mask, outputs_query, outputs_embed = \
                            self.forward_prediction_heads(tgt, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                    predictions_class.append(outputs_class)
                    predictions_mask.append(outputs_mask)
                    predictions_query.append(outputs_query)
                    predictions_embed.append(outputs_embed)
                else:
                    tgt = residual
                    attn_mask = self.prepare_attn_mask(predictions_mask[-1], attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                    predictions_class.append(predictions_class[-1])
                    predictions_mask.append(predictions_mask[-1])
                    predictions_query.append(predictions_query[-1])
                    predictions_embed.append(predictions_embed[-1])

        # get pred_bbox here
        # h, w = outputs_mask.shape[-2:]
        # pred_bboxes = BitMasks(outputs_mask.flatten(0, 1) > 0).get_bounding_boxes().to(attn_mask.device).tensor.reshape(
        #     *outputs_mask.shape[:2], 4)
        # bbox_scale = pred_bboxes.new_tensor([w, h, w, h])
        # pred_bboxes = pred_bboxes / bbox_scale

        out = {
            'pred_logits':  predictions_class[-1],
            'pred_masks':   predictions_mask[-1],
            'pred_queries': predictions_query[-1],
            'pred_embeds':  predictions_embed[-1],
            # 'pred_bboxes': pred_bboxes,
            # 'query_pos_embeds': self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1),
            'aux_outputs': self._set_aux_loss(
                predictions_class[1:], predictions_mask[1:], predictions_query[1:], predictions_embed[1:]
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class  = self.class_embed(decoder_output)
        mask_embed     = self.mask_embed(decoder_output)

        # for track
        reid_embed = self.reid_embed(decoder_output)

        # NOTE: (360 input ytvis2019) mask matmul: 6.12~6.38e-05s
        outputs_mask   = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = self.prepare_attn_mask(outputs_mask, attn_mask_target_size)

        return outputs_class, outputs_mask, attn_mask, decoder_output, reid_embed

    def prepare_attn_mask(self, outputs_mask, attn_mask_target_size):
        # NOTE: (360 input ytvis2019) mask resize: 2.8~4.1e-05s, mask set: ≈8.48e-05s
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        # NOTE: repeat: 4.24e-05s (torch) vs 3.04e-05s (einops), setting mask: 3.36e-05s
        # attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        # attn_mask = einops.repeat(attn_mask, 'b q h w -> (b d) q (h w)', d=self.num_heads) < 0.
        B, Q, H, W = attn_mask.shape
        attn_mask = attn_mask.view(B, Q, -1).unsqueeze(1).expand(-1, self.num_heads, -1, -1).flatten(0, 1) < 0.
        attn_mask[attn_mask.all(dim=-1)] = False
        attn_mask = attn_mask.detach()

        return attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_queries, outputs_embeds):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a,  "pred_masks": b,
             "pred_queries": c, "pred_embeds": d}
            for a, b, c, d in zip(
                outputs_class, outputs_seg_masks,
                outputs_queries, outputs_embeds)
        ]
