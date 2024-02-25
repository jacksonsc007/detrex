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
                operation_order=("encoder_cross_attn", "norm", "ffn", "norm"),
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
        self.use_sparse_key = (self.topk_ratio < 1 ) and (self.topk_ratio > 0)

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
        encoder_key = x
        encoder_value = x
        encoder_key_pos = pos_embed
        query_key_padding_mask = mask
        for layer_id in range(self.num_layers):
            masks.append(mask)
            x, hs, r = self.cascade_layer(x, query_embed, pos_embed, query_key_padding_mask, layer_id, hs, 
                                          encoder_key=encoder_key, encoder_value=encoder_value, encoder_key_pos=encoder_key_pos) # hs also refers to hidden states.
            assert (hs.isfinite().all())

            # the last layer don't need to re-mask
            if self.use_sparse_key and layer_id != (self.num_layers - 1):
                r_d = r.detach()
                hs_d = hs.detach().transpose(0, 1)
                reference_before_sigmoid = inverse_sigmoid(r_d)
                tmp = self.bbox_embed[layer_id](hs_d)
                tmp[..., :2] += reference_before_sigmoid
                outputs_coord = tmp.sigmoid() # (bs, num_q, 4)
                outputs_class = self.class_embed[layer_id](hs_d).sigmoid() # (bs, num_q, 20)
            
                extra_mask = self.mask_from_decoder(outputs_coord, outputs_class, img_true_sizes, img_batched_sizes, (h, w), topk, valid_size)
                object_mask = ( extra_mask.view(bs, -1) | padding_mask).int() # only 0 | 0 get 0 (valid)
                obj_num = object_mask.sum(dim=1) # object tokens for imgs in a batch
                obj_max_num = max(obj_num) # get the max value for batch operation
                assert x.size() == torch.Size([h*w, bs, c])

                obj_idx = torch.topk(object_mask, obj_max_num, 1)[1] # (bs, obj_max_num)
                sparse_key = x.gather(dim=0, index=obj_idx.permute(1, 0)[:, :, None].repeat(1, 1, c)) # (obj_max_num, bs, c)
                sparse_key_pos = pos_embed.gather(dim=0, index=obj_idx.permute(1, 0)[:, :, None].repeat(1, 1, c)) # (obj_max_num, bs, c)
                # fill padding locations with 0
                for img_id in range(bs):
                    valid_num = obj_num[img_id]
                    sparse_key[valid_num: , img_id, :] = 0 
                    sparse_key_pos[valid_num: , img_id, :] = 0 
                           
                # todo check gradient
                encoder_key = sparse_key
                encoder_value = encoder_key
                encoder_key_pos = sparse_key_pos
                query_key_padding_mask = None # keys are selected and all valid (invalid have already been set to 0)
            elif not self.use_sparse_key:
                encoder_key = x
                encoder_value = x
                encoder_key_pos = pos_embed

            hidden_states.append(hs)
            references.append(r)
        hidden_states = torch.stack(hidden_states).transpose(1,2)
        references = torch.stack(references)
        references = references[0] # conditional detr use the same references point.
        #exit() # check dim

        return hidden_states, references

    def mask_from_decoder(self, output_coord, output_class, img_true_size, img_batched_sizes, feature_size, topk, valid_feature_size):
        """
        img_true_size: (bs, 2)
        """
        bs, num_queries, _= output_coord.shape
        assert topk < num_queries
        batched_h, batched_w = img_batched_sizes
        output_class = output_class.max(dim=-1)[0] # (bs, num_query)
        topkid = output_class.topk(topk, dim= -1)[1] # (bs, topk)
        topk_boxes = torch.gather(output_coord, 1, topkid[..., None].repeat(1, 1, 4)) # (bs, topk, 4)
        topk_boxes = box_cxcywh_to_xyxy(topk_boxes) # (bs, topk, 4)
        # valid_feature_size = valid_feature_size.repeat(1, 2).view(bs, 1, 4)
        img_true_size = img_true_size.repeat(1,2).view(bs, 1, 4)
        box_range = img_true_size * topk_boxes # (bs, topk, 4) . (x1, y1, x2, y2)

        grid_y, grid_x = torch.meshgrid(torch.arange(batched_h), torch.arange(batched_w))

        grid_y = grid_y.view(1 , batched_h, batched_w, 1).to(topk_boxes.device)
        grid_x = grid_x.view(1 , batched_h, batched_w, 1).to(topk_boxes.device)
        box_range = box_range.view(bs, 1, 1, topk, 4)
        x1, y1, x2, y2 = torch.split(box_range, 1, -1)
        x1 = x1.squeeze(-1)
        y1 = y1.squeeze(-1)
        x2 = x2.squeeze(-1)
        y2 = y2.squeeze(-1)

        if_in_box = ( grid_x > x1 ) & \
                    ( grid_x < x2 ) & \
                    ( grid_y > y1 ) & \
                    ( grid_y < y2 )
        if_in_any_box = if_in_box.any(dim=-1) # (bs, batched_h, batched_w)
        mask = (~if_in_any_box).float() # 0 means valid
        mask = F.interpolate(mask[:, None], feature_size).bool().squeeze(1)
        return mask

    def cascade_layer(self, x, query_embed, pos_embed, query_key_padding_mask, layer_id, hidden_state, 
                      encoder_key, encoder_value, encoder_key_pos):
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
        memory = enc_layer(x, key=encoder_key, value=encoder_value, key_pos=encoder_key_pos, query_pos=pos_embed, query_key_padding_mask=query_key_padding_mask)
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
