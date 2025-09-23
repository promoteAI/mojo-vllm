import torch
from torch import nn
import triton
import triton.language as tl

from mojovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables,
                                       num_heads=self.num_heads)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True, num_heads=self.num_heads)
        return o


def _gather_seq_from_cache(cache: torch.Tensor, block_table_row: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """
    Gather a contiguous sequence [T, H_kv, D] from KV cache using block_table for one sequence.
    cache: [num_blocks_total, block_size, H_kv, D]
    block_table_row: [num_blocks_for_seq] (padded with -1)
    total_tokens: seqlen to gather
    """
    # Infer block_size
    block_size = cache.size(1)
    gathered = []
    remaining = int(total_tokens)
    for block_id in block_table_row.tolist():
        if block_id < 0 or remaining <= 0:
            break
        take = block_size if remaining > block_size else remaining
        block = cache[block_id]
        if take == block_size:
            gathered.append(block)
        else:
            gathered.append(block[:take])
        remaining -= take
    if gathered:
        return torch.cat(gathered, dim=0)
    else:
        # No prefix blocks; total_tokens may be 0
        return cache.new_empty((0, cache.size(2), cache.size(3)))


def _expand_kv_heads_to_q_heads(x: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """
    x: [T, H_kv, D] -> [T, H_q, D] by repeating heads evenly.
    """
    num_kv_heads = x.size(1)
    if num_kv_heads == num_query_heads:
        return x
    group = num_query_heads // num_kv_heads
    return x.repeat_interleave(group, dim=1)


@torch.no_grad()
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    block_table: torch.Tensor | None = None,
    num_heads: int | None = None,
):
    """
    Reference implementation of varlen attention without flash-attn.
    Inputs follow packed layout with cu_seqlens_*; returns packed outputs [sum(T_q), H_q, D].
    If block_table is provided, k and v are KV caches of shape [B_total_blocks, block_size, H_kv, D].
    """
    device = q.device
    dtype = q.dtype
    B = cu_seqlens_q.numel() - 1
    outputs = torch.empty_like(q)

    # For each sequence, materialize K/V if needed, then compute attention.
    for b in range(B):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        k_start = int(cu_seqlens_k[b].item())
        k_end = int(cu_seqlens_k[b + 1].item())
        q_seq = q[q_start:q_end]  # [Tq, H_q, D]

        if block_table is None:
            k_seq = k[k_start:k_end]
            v_seq = v[k_start:k_end]
        else:
            total_k = k_end - k_start
            bt_row = block_table[b]
            k_seq = _gather_seq_from_cache(k, bt_row, total_k)
            v_seq = _gather_seq_from_cache(v, bt_row, total_k)

        # Expand KV heads to query heads if needed
        H_q = q_seq.size(1)
        k_seq = _expand_kv_heads_to_q_heads(k_seq, H_q)
        v_seq = _expand_kv_heads_to_q_heads(v_seq, H_q)

        # Compute attention per head: reshape to [H, T, D]
        qh = q_seq.transpose(0, 1)  # [H, Tq, D]
        kh = k_seq.transpose(0, 1)  # [H, Tk, D]
        vh = v_seq.transpose(0, 1)  # [H, Tk, D]
        attn_scores = torch.matmul(qh, kh.transpose(-1, -2)) * softmax_scale  # [H, Tq, Tk]
        if causal:
            Tq = qh.size(1)
            Tk = kh.size(1)
            # Create causal mask allowing cross prefix: only enforce i < j mask within each sequence window
            mask = torch.ones((Tq, Tk), device=device, dtype=torch.bool)
            mask = torch.triu(mask, diagonal=1)
            attn_scores.masked_fill_(mask, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype)
        oh = torch.matmul(attn_probs, vh)  # [H, Tq, D]
        outputs[q_start:q_end] = oh.transpose(0, 1)

    return outputs


@torch.no_grad()
def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    num_heads: int | None = None,
):
    """
    Decode-time attention using KV cache without flash-attn.
    q: [B, 1, H_q, D]; returns [B, H_q, D]
    k_cache/v_cache: [num_blocks_total, block_size, H_kv, D]
    cache_seqlens: [B] context lengths including current token position
    block_table: [B, num_blocks_for_seq]
    """
    assert q.dim() == 4 and q.size(1) == 1
    B = q.size(0)
    H_q = q.size(2)
    D = q.size(3)
    outputs = torch.empty((B, H_q, D), device=q.device, dtype=q.dtype)

    for b in range(B):
        T = int(cache_seqlens[b].item())
        bt_row = block_table[b]
        k_seq = _gather_seq_from_cache(k_cache, bt_row, T)
        v_seq = _gather_seq_from_cache(v_cache, bt_row, T)
        k_seq = _expand_kv_heads_to_q_heads(k_seq, H_q)
        v_seq = _expand_kv_heads_to_q_heads(v_seq, H_q)

        q_b = q[b, 0]  # [H_q, D]
        kh = k_seq.transpose(0, 1)  # [H_q, T, D]
        vh = v_seq.transpose(0, 1)  # [H_q, T, D]
        scores = torch.matmul(q_b.unsqueeze(1), kh.transpose(-1, -2)).squeeze(1) * softmax_scale  # [H_q, T]
        if causal:
            # Only attend up to the last position (T-1) when predicting next
            # q corresponds to position T-1
            pass  # scores already only over existing tokens
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs.unsqueeze(1), vh).squeeze(1)  # [H_q, D]
        outputs[b] = out

    return outputs