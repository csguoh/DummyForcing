import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import os



def online_head_classification(query, key, ar_start):
    B, L, head, C = query.shape
    HW = L//3 if ar_start ==1 else L
    num_sampled_rows = HW // 3
    sampled_rows = torch.randint(low=0, high=L, size=(num_sampled_rows,))
    sampled_q = query[:, sampled_rows]
    sampled_q = sampled_q.transpose(1, 2) # B, head, L, C
    key = key.transpose(1, 2) # B, head, 3L, C
    sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (C ** 0.5)
    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1) # B, head, L, 3L
    last_chunk_agg = sampled_attn_weights[:, :, :, -L:].sum(dim=-1).mean(dim=-1)
    mid_chunk_agg = sampled_attn_weights[:, :, :, HW:-L].sum(dim=-1).mean(dim=-1)
    first_chunk_agg = sampled_attn_weights[:, :, :, :HW].sum(dim=-1).mean(dim=-1)
    total_chunk_agg = torch.stack([first_chunk_agg, mid_chunk_agg, last_chunk_agg]) # 3, B, head
    return total_chunk_agg


def dynamic_head_programming(probs, num_dummy=180):
    """
    probs: [num_layer, num_head, 3] tensor
    num_dummy: target number of elements in group C
    layer_threshold: layers below this use weight 1.0
    weight_after: weight multiplier for layers >= threshold
    Returns: three dicts {layer_idx: [head_indices]} for groups A, B, C
    """
    num_layer, num_head, _ = probs.shape
    p0_flat = probs[:, :, 0].reshape(-1)
    p1_flat = probs[:, :, 1].reshape(-1)
    p0_norm = p0_flat  / p0_flat.sum()
    p1_norm = p1_flat  / p1_flat.sum()
    cost = torch.maximum(p0_norm, p1_norm)
    sorted_indices = torch.argsort(cost)
    c_indices_flat = sorted_indices[:num_dummy]
    assignment = torch.zeros(num_layer * num_head, dtype=torch.long)
    assignment[c_indices_flat] = 2
    remaining_mask = assignment != 2
    remaining_indices = torch.nonzero(remaining_mask, as_tuple=True)[0]

    for idx in remaining_indices:
        if p0_norm[idx] < p1_norm[idx]:
            assignment[idx] = 1
        else:
            assignment[idx] = 0

    assignment = assignment.reshape(num_layer, num_head)
    group_a = {}
    group_b = {}
    group_c = {}
    for layer_idx in range(num_layer):
        group_a[layer_idx] = (assignment[layer_idx] == 0).nonzero(as_tuple=True)[0].tolist()
        group_b[layer_idx] = (assignment[layer_idx] == 1).nonzero(as_tuple=True)[0].tolist()
        group_c[layer_idx] = (assignment[layer_idx] == 2).nonzero(as_tuple=True)[0].tolist()
    return group_a, group_b, group_c



def heterogeneous_memory_allocation(global_kv_cache, num_dummy=180):
    global_frame_attn_score = torch.stack([layer_info['frame_attn_score'][:,0] for layer_info in global_kv_cache]).transpose(1,2)
    global_group_first, global_group_mid, global_group_last = dynamic_head_programming(global_frame_attn_score, num_dummy)
    for layer_idx in range(len(global_kv_cache)):
        group_first, group_mid, group_last = global_group_first[layer_idx], global_group_mid[layer_idx], global_group_last[layer_idx]
        cur_cache = global_kv_cache[layer_idx]
        HW = cur_cache['sink_k'].shape[1]
        cur_cache['sink_k'] =  torch.cat([cur_cache['sink_k'][:, :, group_first], cur_cache['local_k'][:,-HW:,group_last]], dim=2).contiguous().clone()
        cur_cache['sink_v'] =  torch.cat([cur_cache['sink_v'][:, :, group_first], cur_cache['local_v'][:,-HW:,group_last]], dim=2).contiguous().clone()
        cur_cache['local_k'] = cur_cache['local_k'][:, :, group_mid].contiguous().clone()
        cur_cache['local_v'] = cur_cache['local_v'][:, :, group_mid].contiguous().clone()
        cur_cache['headgroup_first'] = group_first
        cur_cache['headgroup_mid'] = group_mid
        cur_cache['headgroup_last'] = group_last