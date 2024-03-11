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

import math
import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid


class DNDeformableDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        operation_order: tuple = ("self_attn", "norm", "ffn", "norm"),
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
    ):
        super(DNDeformableDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=operation_order,
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.num_layers = len(self.layers)

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

        for layer_idx in range(self.num_layers):
            query = self.cascade_stage_encoder_part(
                layer_idx,
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
        return query
    
    def cascade_stage_encoder_part(
        self,
        layer_idx,
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
        layer = self.layers[layer_idx]
        assert kwargs.get("reference_points", None) is not None
        query = layer(
            query,
            key,
            value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        # post-norm for last encoder layer
        if (layer_idx == self.num_layers - 1 ) and ( self.post_norm_layer is not None ):
            query = self.post_norm_layer(query)
        return query



class DNDeformableDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
    ):
        super(DNDeformableDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None
        self.class_embed = None
        self.num_layers = len(self.layers)

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
        reference_points=None,  # num_queries, 4
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        # reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4

        intermediate = []
        intermediate_reference_points = []
        for layer_idx in range(len(self.layers)):
            output, reference_points = self.cascade_stage_decoder_part(
                layer_idx,
                output,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                reference_points,
                valid_ratios,
                **kwargs
            )
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

    def cascade_stage_decoder_part(
        self,
        layer_idx,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,  # num_queries, 4
        valid_ratios=None,
        **kwargs
    ):
        output = query
        if reference_points.shape[-1] == 4:
            reference_points_input = (
                reference_points[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            )
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

        query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
        raw_query_pos = self.ref_point_head(query_sine_embed)
        pos_scale = self.query_scale(output) if layer_idx != 0 else 1
        query_pos = pos_scale * raw_query_pos

        layer = self.layers[layer_idx]
        output, decoder_sampling_locations, decoder_attention_weights = layer(
            output,
            key,
            value,
            query_pos=query_pos,
            key_pos=key_pos,
            query_sine_embed=query_sine_embed,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points_input,
            **kwargs,
        )

        if self.bbox_embed is not None:
            tmp = self.bbox_embed[layer_idx](output)
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            else:
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()
        return output, reference_points, decoder_sampling_locations, decoder_attention_weights



class DNDeformableDetrTransformer(nn.Module):
    def __init__(
        self,
        encoder=None,
        decoder=None,
        as_two_stage=False,
        num_feature_levels=4,
        two_stage_num_proposals=300,
    ):
        super(DNDeformableDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
            self.enc_outpout_norm = nn.LayerNorm(self.embed_dim)
            self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

        self.num_enc_layers = self.encoder.num_layers
        self.num_dec_layers = self.decoder.num_layers
        assert self.num_enc_layers == self.num_dec_layers
        self.num_stages = self.encoder.num_layers


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            wh = torch.ones_like(ref) * 0.05 
            # wh = torch.ones_like(ref) * 0.05 * (2.0**lvl) # relative to real size
            ref = torch.cat([ref, wh], dim=2)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # convert normalized value relative to real image size to padded image size 
        reference_points = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None] # (bs,all_lvl_num, 1, 4) * (bs, 1, num_lvl, 4)
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        input_label_query,
        input_box_query,
        attn_masks,
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        fixed_encoder_reference_boxes = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        decoder_reference_points = None
        decoder_query = None
        memory = feat_flatten
        encoder_reference_points = fixed_encoder_reference_boxes

        dec_inter_states = []
        dec_inter_references = []
        enc_inter_outputs_class = []
        enc_inter_outputs_coord_unact = []
        init_dec_refs = []
        for stage_id in range(self.num_stages):
            memory, decoder_query, decoder_reference_points, \
            enc_output_cls, enc_output_coord, init_dec_ref, \
            decoder_sampling_locations, decoder_attention_weights= self.cascade_stage(
                stage_id=stage_id,
                encoder_query=memory,
                encoder_key=None,
                encoder_value=None,
                encoder_query_pos=lvl_pos_embed_flatten,
                encoder_key_pos=None,
                encoder_attn_masks=None,
                encoder_query_key_padding_mask=mask_flatten,
                encoder_key_padding_mask=None,
                encoder_reference_points=encoder_reference_points,
                # ------------------------------------------------
                decoder_query=decoder_query,
                decoder_query_pos=None,
                decoder_key_pos=None,
                decoder_attn_masks=attn_masks,
                decoder_query_key_padding_mask=None,
                decoder_key_padding_mask=mask_flatten,
                decoder_reference_points=decoder_reference_points,
                # ------------------------------------------------
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                # ------------------------------------------------
                input_box_query=input_box_query,
                input_label_query=input_label_query,
                **kwargs
            )

            bs, n_all_q, c = decoder_query.size()
            decoder_matching_query = decoder_query.clone().detach()[:, self.num_noised_queries:]
            decoder_matching_query_rf = decoder_reference_points.clone().detach()[:, self.num_noised_queries:]
            N, Len_q, n_heads, n_levels, n_points, _ = decoder_sampling_locations.size()
            sampling_locations = decoder_sampling_locations.clone().detach()[:, self.num_noised_queries:].unsqueeze(1) # adds layer dim
            attention_weights = decoder_attention_weights.clone().detach()[:, self.num_noised_queries:].unsqueeze(1)

            decoder_matching_query_cls = self.decoder.class_embed[stage_id](decoder_matching_query).sigmoid().max(-1)[0].detach()
            self.obj_thr = 0.2
            # (N, num_matching_queries) -> (N, num_matching_queries, n_heads, n_levels, n_points)
            # -> (N, 1, num_matching_queries, n_heads, n_levels, n_points)
            dec_obj_mask = (decoder_matching_query_cls > self.obj_thr).view(N, self.num_matching_queries, 1, 1, 1).repeat(1, 1, n_heads, n_levels, n_points)
            dec_obj_mask = dec_obj_mask[:, None]
            attention_weights = attention_weights.masked_fill(~dec_obj_mask, 0) # empty the weights of non-object queries
            
            # (bs, num_all_lvl_tokens, num_matching_queries)
            decoder_cross_attention_map =attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attention_weights)

            max_attn_weight, max_query_idx = decoder_cross_attention_map.max(dim=2)
            # max_attn_weight2, max_query_idx2 = decoder_cross_attention_map[:, :, :self.num_queries_one2one].max(dim=2)
            new_enc_refs = []
            self.attn_weight_thr = 0.1
            for img_id in range(N):
                object_token_idx = (max_attn_weight[img_id] > self.attn_weight_thr).nonzero().squeeze(1)
                # object_token_idx2 = (max_attn_weight2[img_id] > 0).nonzero().squeeze(1)

                # valid_ratio1 = len(object_token_idx) / num_tokens_all_lvl
                # valid_ratio2 = len(object_token_idx2) / num_tokens_all_lvl
                    
                if len(object_token_idx) !=0:
                    valid_obj_query_idx = (max_query_idx[img_id])[object_token_idx]
                    # encoder_reference_boxes[0]: (num_all_lvl_tokens, num_levels, 4)
                    # decoder_reference_points: (N, Len_q, 4)
                    per_img_dec_ref_box = (decoder_matching_query_rf[img_id]).unsqueeze(dim=1).repeat(1, n_levels, 1) # (Len_q, n_levels, 4)
                    # convert ref from real image size to padded image size
                    valid_ratio_per = valid_ratios[img_id]
                    per_img_dec_ref_box = per_img_dec_ref_box * (torch.cat([valid_ratio_per, valid_ratio_per], -1))[None] # (Len_q, n_levels, 4) * (1, n_levels, 4)
                    new_enc_refs.append( 
                        fixed_encoder_reference_boxes[img_id].scatter(
                        dim=0, 
                        index=object_token_idx[:, None, None].repeat(1, n_levels, 4), # (num_object_token, n_levels, 4)
                        src=per_img_dec_ref_box[valid_obj_query_idx]
                        )
                    )
                else:
                    new_enc_refs.append(fixed_encoder_reference_boxes[img_id])
            new_enc_refs = torch.stack(new_enc_refs, dim=0) # (N, num_all_lvl_tokens, n_levels, 4)
            encoder_reference_points = new_enc_refs.unsqueeze(3)

            
            # get the results of each stage
            dec_inter_states.append(decoder_query)
            dec_inter_references.append(decoder_reference_points)
            enc_inter_outputs_class.append(enc_output_cls)
            enc_inter_outputs_coord_unact.append(enc_output_coord)
            init_dec_refs.append(init_dec_ref)
        dec_inter_references = torch.stack(dec_inter_references)
        dec_inter_states = torch.stack(dec_inter_states)
        init_reference_out = init_dec_refs[0]
        assert init_reference_out is not None
        enc_outputs_class = enc_inter_outputs_class[0]
        enc_outputs_coord_unact = enc_inter_outputs_coord_unact[0]

        if self.as_two_stage:
            return (
                dec_inter_states,
                init_reference_out,
                dec_inter_references,
                enc_outputs_class, # refers to first stage encoder output
                enc_outputs_coord_unact,
            )
        return dec_inter_states, init_reference_out, dec_inter_references, None, None


    def cascade_stage(
        self,
        stage_id,
        encoder_query,
        encoder_key,
        encoder_value,
        encoder_query_pos,
        encoder_key_pos,
        encoder_attn_masks,
        encoder_query_key_padding_mask,
        encoder_key_padding_mask,
        encoder_reference_points,
        decoder_query,
        decoder_query_pos,
        decoder_key_pos,
        decoder_attn_masks,
        decoder_query_key_padding_mask,
        decoder_key_padding_mask,
        decoder_reference_points,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        input_box_query,
        input_label_query,
        **kwargs
    ):

        memory, _, _ = self.encoder.cascade_stage_encoder_part(
            layer_idx=stage_id,
            query=encoder_query,
            key=encoder_key,
            value=encoder_value,
            query_pos=encoder_query_pos,
            key_pos=encoder_key_pos,
            attn_masks=encoder_attn_masks,
            query_key_padding_mask=encoder_query_key_padding_mask,
            key_padding_mask=encoder_key_padding_mask,
            spatial_shapes=spatial_shapes,
            reference_points=encoder_reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        # we only use two stage for the 1st stage #TODO might improve
        init_reference_out = None
        enc_outputs_class = None
        enc_outputs_coord_unact = None
        if stage_id == 0:
            assert decoder_reference_points is None
            if self.as_two_stage:
                assert input_box_query is None, "query_embed should be None in two-stage"
                output_memory, output_proposals = self.gen_encoder_output_proposals(
                    memory, encoder_query_key_padding_mask, spatial_shapes
                )
                # output_memory: bs, num_tokens, c
                # output_proposals: bs, num_tokens, 4. unsigmoided.
                # output_proposals: bs, num_tokens, 4

                enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
                enc_outputs_coord_unact = (
                    self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
                )  # unsigmoided.

                topk = self.two_stage_num_proposals
                topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

                # extract region proposal boxes
                topk_coords_unact = torch.gather(
                    enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
                )  # unsigmoided.
                reference_points = topk_coords_unact.detach().sigmoid()
                init_reference_out = reference_points

                # extract region features
                target_unact = torch.gather(
                    output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
                )
                target = target_unact.detach()
            else:
                reference_points = input_box_query.sigmoid()
                target = input_label_query
                init_reference_out = reference_points
            decoder_reference_points = reference_points
            decoder_query = target
        assert decoder_reference_points is not None
        output, new_reference_points, \
            decoder_sampling_locations, decoder_attention_weights = self.decoder.cascade_stage_decoder_part(
            layer_idx=stage_id,
            query=decoder_query,
            key=memory,
            value=memory,
            query_pos=decoder_query_pos,
            key_pos=decoder_key_pos,
            attn_masks=decoder_attn_masks,
            query_key_padding_mask=decoder_query_key_padding_mask,
            key_padding_mask=decoder_key_padding_mask,
            reference_points=decoder_reference_points,
            valid_ratios=valid_ratios,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        return memory, output, new_reference_points, enc_outputs_class, enc_outputs_coord_unact, init_reference_out, \
                decoder_sampling_locations, decoder_attention_weights

def attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations, attention_weights):
    # sampling_locations: [N, n_layers, Len_q, n_heads, n_levels, n_points, 2]
    # attention_weights: [N, n_layers, Len_q, n_heads, n_levels, n_points]
    N, n_layers, Len_q, n_heads, *_ = sampling_locations.shape
    sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 3)
    # [N * n_layers * n_heads * Len_q, n_points, n_levels, 2]
    attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 3)
    # [N * n_layers * n_heads * Len_q, n_points, n_levels]

    rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1) # hw -> wh (xy)
    col_row_float = sampling_locations * rev_spatial_shapes # [N * n_layers * n_heads * Len_q, n_points, n_levels, 2]
    # get 4 corner integeral positions around the floating-type sampling locations. 
    col_row_ll = col_row_float.floor().to(torch.int64) # [N * n_layers * n_heads * Len_q, n_points, n_levels, 2]
    zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device) # [N * n_layers * n_heads * Len_q, n_points, n_levels, 2]
    one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device) # [N * n_layers * n_heads * Len_q, n_points, n_levels, 2]
    col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)
    col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)
    col_row_hh = col_row_ll + 1
    # compute magin for bilinear interpolation
    margin_ll = (col_row_float - col_row_ll).prod(dim=-1)
    margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
    margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
    margin_hh = (col_row_float - col_row_hh).prod(dim=-1) # [N * n_layers * n_heads * Len_q, n_points, n_levels, 2]

    flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1]))) # [N * n_layers * n_heads * Len_q, num_all_lvl_tokens]
    flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device) # [N * n_layers * n_heads * Len_q, num_all_lvl_tokens]

    zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
    for col_row, margin in zipped:
        valid_mask = torch.logical_and(
            torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
            torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
        )
        #  [N * n_layers * n_heads * Len_q, n_points, n_levels] * [n_levels, ] + 
        #  [N * n_layers * n_heads * Len_q, n_points, n_levels] + [n_levels]
        idx = col_row[..., 1] * spatial_shapes[..., 1] + col_row[..., 0] + level_start_index
        idx = (idx * valid_mask).flatten(1, 2)
        weights = (attention_weights * valid_mask * margin).flatten(1)
        flat_grid.scatter_add_(1, idx, weights)

    return flat_grid.reshape(N, Len_q, n_layers, n_heads, -1).sum((2,3)).permute(0, 2, 1)