# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from detrex.utils.misc import inverse_sigmoid

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    MultiheadAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
    box_cxcywh_to_xyxy
)

INK_INF = 1e20

class ConditionalDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.PReLU(),
        post_norm: bool = False,
        num_layers: int = 6,
        batch_first: bool = False,
    ):
        super(ConditionalDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(normalized_shape=embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class ConditionalDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        num_layers: int = None,
        batch_first: bool = False,
        post_norm: bool = True,
        return_intermediate: bool = True,
    ):
        super(ConditionalDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    ConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                    ConditionalCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(self.embed_dim, self.embed_dim, 2, 2)

        self.bbox_embed = None

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(
            query_pos
        )  # [num_queries, batch_size, 2]
        reference_points: torch.Tensor = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for idx, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)  # [num_queries, batch_size, 2]

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # get sine embedding for the query vector
            query_sine_embed = get_sine_pos_embed(obj_center)
            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            query: torch.Tensor = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)

        if self.return_intermediate:
            return [
                torch.stack(intermediate).transpose(1, 2),
                reference_points,
            ]

        return query.unsqueeze(0)


class ConditionalDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None, topk_ratio=0.5):
        super(ConditionalDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim
        self.encoder_layer = self.encoder.num_layers
        self.decoder_layer = self.decoder.num_layers
        assert self.encoder_layer == self.decoder_layer
        self.num_layers = self.encoder_layer
        self.init_weights()

        self.topk_ratio = topk_ratio

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed, img_true_sizes, img_batched_sizes):
        """
        
        x: 
            feature from backbone. ( bs, c, h, w)
        mask: 
            feature mask for batched images. (bs, h, w)
        query_embed: 
            learnable position query embedding for decoder.
        pos_embed: 
            positional embeddng for x.
        img_true_size: 
            valid images content size.
        img_batched_size: 
            batched images have the same shape but different valid content.
        """
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        valid_h = (~mask).cumsum(dim=1)[:, -1, 0]
        valid_w = (~mask).cumsum(dim=2)[:, 0, -1]
        valid_size = torch.stack([valid_h, valid_w], dim = -1)
        mask = mask.view(bs, -1)


        hidden_states = []
        references = []
        masks = []
        hs = torch.zeros_like(query_embed) # initial content query for decoder
        topk = int( self.num_queries * self.topk_ratio )
        padding_mask = mask
        for layer_id in range(self.num_layers):
            masks.append(mask)
            x, hs, r = self.cascade_layer(x, query_embed, pos_embed, mask, layer_id, hs) # hs also refers to hidden states.
            assert (hs.isfinite().all())
            # the last layer don't need to re-mask
            if layer_id != (self.num_layers - 1):
                r_d = r.detach()
                hs_d = hs.detach().transpose(0, 1)
                reference_before_sigmoid = inverse_sigmoid(r_d)
                tmp = self.bbox_embed[layer_id](hs_d)
                tmp[..., :2] += reference_before_sigmoid
                outputs_coord = tmp.sigmoid() # (bs, num_q, 4)
                outputs_class = self.class_embed[layer_id](hs_d).sigmoid() # (bs, num_q, 20)
            
                extra_mask = self.mask_from_decoder(outputs_coord, outputs_class, img_true_sizes, img_batched_sizes, (h, w), topk, valid_size)
                mask = ( extra_mask.view(bs, -1) | padding_mask) # only 0 | 0 get 0 (valid)
                mask = mask.float().masked_fill(mask, -INK_INF) # avoid to generate nan in the softmax operation in nn.MultiheadAttention.
                
                # mask = mask.view(bs, h, w) 
                # center_y, center_x = torch.unbind((0.5 * valid_size).long(), dim=-1)
                # mask[range(bs), center_y, center_x] = False
                # mask = mask.view(bs, -1)
                assert id(mask) != id(padding_mask)

            hidden_states.append(hs)
            references.append(r)
        hidden_states = torch.stack(hidden_states).transpose(1,2)
        references = torch.stack(references)
        references = references[0] # conditional detr use the same references point.
        #exit() # check dim

        return hidden_states, references

    def mask_from_decoder(self, output_coord, output_class, img_true_size, img_batched_sizes, feature_size, topk, valid_feature_size):
        bs, num_queries, _= output_coord.shape
        assert topk < num_queries
        batched_h, batch_w = img_batched_sizes
        extra_mask = output_coord.new_ones(bs, *feature_size, dtype=torch.bool) # 0 mean valid, 1 means invalid
        output_class = output_class.max(dim=-1)[0] # (bs, num_query)
        topkid = output_class.topk(topk, dim= -1)[1] # (bs, topk)
        topk_boxes = torch.gather(output_coord, 1, topkid[..., None].repeat(1, 1, 4)) # (bs, topk, 4)
        topk_boxes = box_cxcywh_to_xyxy(topk_boxes)
        valid_feature_size = valid_feature_size.repeat(1, 2).view(bs, 1, 4)

        # we omit the difference between true size and batched size here
        box_feature_range = ( topk_boxes * valid_feature_size ).floor().to(torch.int64)
        assert  (box_feature_range[..., :2] <= box_feature_range[..., 2:]).all()

        for img_idx in range(bs):
            for loc_id in range(topk):
                lx, ly, rx, ry = box_feature_range[img_idx, loc_id]
                extra_mask[img_idx][ly:ry, lx:rx] = False
       
        
        return extra_mask

    def cascade_layer(self, x, query_embed, pos_embed, mask, layer_id, hidden_state):
        """
        x:
            the enhanced backbone feature
        query_embed: 
            learnable position query embedding for decoder.
        pos_embed: 
            encoder positional encoding for backbone feature.
        mask:
            mask for backbone feature.
        layer_id:
            current index of encoder and decoder layer.
        hidden_state:
            decoder output for box & class prediction.
        """
        
        enc_layer = self.encoder.layers[layer_id]
        dec_layer = self.decoder.layers[layer_id]

        # get encoder output
        memory = enc_layer(x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask)
        assert memory.isfinite().all()
        if layer_id == self.num_layers: # special treatment for last encoder layer following the original code.
            if self.encoder.post_norm_layer is None:
                memory = self.encoder.post_norm_layer(memory)
        # get decoder output
        dec_query_content = hidden_state
        dec_query_pos = query_embed
        dec_key_content = memory
        dec_key_pos = pos_embed

        reference_points_before_sigmoid = self.decoder.ref_point_head(dec_query_pos)
        reference_points: torch.Tensor = reference_points_before_sigmoid.sigmoid().transpose(0,1)
        obj_center = reference_points[..., :2].transpose(0, 1)
        # do not apply transform in position in the first decoder layer
        if layer_id == 0:
            position_transform = 1
        else:
            position_transform = self.decoder.query_scale(dec_query_content)
        # get sine embedding for the query vector
        query_sine_embed = get_sine_pos_embed(obj_center)
        # apply position transform
        query_sine_embed = query_sine_embed[..., : self.decoder.embed_dim] * position_transform

        hidden_state = dec_layer(
            query=dec_query_content,
            key=dec_key_content,
            value=dec_key_content,
            query_pos=dec_query_pos,           
            key_pos=dec_key_pos,
            query_sine_embed=query_sine_embed,
            is_first_layer=(layer_id == 0),
        )

        if self.decoder.post_norm_layer is not None:
            hidden_state = self.decoder.post_norm_layer(hidden_state)
        
        if (layer_id == self.num_layers) & (self.decoder.post_norm_layer is not None): # another layernorm for the last output
            hidden_state = self.decoder.post_norm_layer(hidden_state)
        

        return memory, hidden_state, reference_points
