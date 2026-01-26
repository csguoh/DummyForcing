import torch
import triton
import triton.language as tl
import time


@triton.jit
def extract_heads_kernel(
        roped_query_ptr, roped_key_ptr, v_ptr,
        q1_ptr, k1_ptr, v1_ptr,
        q2_ptr, k2_ptr, v2_ptr,
        headgroup_last_ptr, headgroup_first_mid_ptr,
        B, L, num_heads, C,
        num_last, num_first_mid,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    b_idx = pid // (L * num_heads)
    remainder = pid % (L * num_heads)
    l_idx = remainder // num_heads
    h_idx = remainder % num_heads

    if b_idx >= B or l_idx >= L or h_idx >= num_heads:
        return

    is_last = 0
    out_h_last = 0
    for i in range(num_last):
        head_id = tl.load(headgroup_last_ptr + i)
        if h_idx == head_id:
            is_last = 1
            out_h_last = i

    is_first_mid = 0
    out_h_first_mid = 0
    for i in range(num_first_mid):
        head_id = tl.load(headgroup_first_mid_ptr + i)
        if h_idx == head_id:
            is_first_mid = 1
            out_h_first_mid = i

    base_offset = b_idx * L * num_heads * C + l_idx * num_heads * C + h_idx * C

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    if is_last == 1:
        q_data = tl.load(roped_query_ptr + base_offset + offsets, mask=mask)
        k_data = tl.load(roped_key_ptr + base_offset + offsets, mask=mask)
        v_data = tl.load(v_ptr + base_offset + offsets, mask=mask)

        out_offset = b_idx * L * num_last * C + l_idx * num_last * C + out_h_last * C
        tl.store(q1_ptr + out_offset + offsets, q_data, mask=mask)
        tl.store(k1_ptr + out_offset + offsets, k_data, mask=mask)
        tl.store(v1_ptr + out_offset + offsets, v_data, mask=mask)

    if is_first_mid == 1:
        q_data = tl.load(roped_query_ptr + base_offset + offsets, mask=mask)
        k_data = tl.load(roped_key_ptr + base_offset + offsets, mask=mask)
        v_data = tl.load(v_ptr + base_offset + offsets, mask=mask)

        out_offset = b_idx * L * num_first_mid * C + l_idx * num_first_mid * C + out_h_first_mid * C
        tl.store(q2_ptr + out_offset + offsets, q_data, mask=mask)
        tl.store(k2_ptr + out_offset + offsets, k_data, mask=mask)
        tl.store(v2_ptr + out_offset + offsets, v_data, mask=mask)



def extract_heads_triton(roped_query, roped_key, v, headgroup_first_mid, headgroup_last):
    B, L, num_heads, C = roped_query.shape

    num_last = len(headgroup_last)
    num_first_mid = len(headgroup_first_mid)

    q1 = torch.empty(B, L, num_last, C, device=roped_query.device, dtype=roped_query.dtype)
    k1 = torch.empty(B, L, num_last, C, device=roped_key.device, dtype=roped_key.dtype)
    v1 = torch.empty(B, L, num_last, C, device=v.device, dtype=v.dtype)

    q2 = torch.empty(B, L, num_first_mid, C, device=roped_query.device, dtype=roped_query.dtype)
    k2 = torch.empty(B, L, num_first_mid, C, device=roped_key.device, dtype=roped_key.dtype)
    v2 = torch.empty(B, L, num_first_mid, C, device=v.device, dtype=v.dtype)

    headgroup_last_tensor = torch.tensor(headgroup_last, device=roped_query.device, dtype=torch.int32)
    headgroup_first_mid_tensor = torch.tensor(headgroup_first_mid, device=roped_query.device, dtype=torch.int32)

    grid = (B * L * num_heads,)
    BLOCK_SIZE = 128

    extract_heads_kernel[grid](
        roped_query, roped_key, v,
        q1, k1, v1,
        q2, k2, v2,
        headgroup_last_tensor, headgroup_first_mid_tensor,
        B, L, num_heads, C,
        num_last, num_first_mid,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q1, k1, v1, q2, k2, v2


def extract_heads_torch(roped_query, roped_key, v, headgroup_first_mid, headgroup_last):
    q1 = roped_query[:, :, headgroup_last, :]
    k1 = roped_key[:, :, headgroup_last, :]
    v1 = v[:, :, headgroup_last, :]

    q2 = roped_query[:, :, headgroup_first_mid, :]
    k2 = roped_key[:, :, headgroup_first_mid, :]
    v2 = v[:, :, headgroup_first_mid, :]

    return q1, k1, v1, q2, k2, v2


def profile():
    device = 'cuda'
    B, L, num_heads, C = 1, 4680, 12, 128

    headgroup_last = [0, 2, 6, 7, 8, 10, 11]
    headgroup_first_mid = [1, 3, 4, 5, 9]

    roped_query = torch.randn(B, L, num_heads, C, device=device, dtype=torch.float16)
    roped_key = torch.randn(B, L, num_heads, C, device=device, dtype=torch.float16)
    v = torch.randn(B, L, num_heads, C, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = extract_heads_torch(roped_query, roped_key, v, headgroup_first_mid, headgroup_last)
        _ = extract_heads_triton(roped_query, roped_key, v, headgroup_first_mid, headgroup_last)

    torch.cuda.synchronize()

    # Benchmark torch
    num_runs = 100
    start = time.perf_counter()
    torch.cuda.synchronize()

    for _ in range(num_runs):
        q1_t, k1_t, v1_t, q2_t, k2_t, v2_t = extract_heads_torch(roped_query, roped_key, v, headgroup_first_mid,
                                                                 headgroup_last)

    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) * 1000 / num_runs

    # Benchmark triton
    start = time.perf_counter()
    torch.cuda.synchronize()

    for _ in range(num_runs):
        q1_tr, k1_tr, v1_tr, q2_tr, k2_tr, v2_tr = extract_heads_triton(roped_query, roped_key, v, headgroup_first_mid,
                                                                        headgroup_last)

    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) * 1000 / num_runs

    # Verify correctness
    q1_t, k1_t, v1_t, q2_t, k2_t, v2_t = extract_heads_torch(roped_query, roped_key, v, headgroup_first_mid,
                                                             headgroup_last)
    q1_tr, k1_tr, v1_tr, q2_tr, k2_tr, v2_tr = extract_heads_triton(roped_query, roped_key, v, headgroup_first_mid,
                                                                    headgroup_last)

    print(f"Torch indexing time: {torch_time:.3f} ms")
    print(f"Triton kernel time: {triton_time:.3f} ms")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    print(f"\nCorrectness check:")
    print(f"q1 match: {torch.allclose(q1_t, q1_tr, atol=1e-5)}")
    print(f"k1 match: {torch.allclose(k1_t, k1_tr, atol=1e-5)}")
    print(f"v1 match: {torch.allclose(v1_t, v1_tr, atol=1e-5)}")
    print(f"q2 match: {torch.allclose(q2_t, q2_tr, atol=1e-5)}")
    print(f"k2 match: {torch.allclose(k2_t, k2_tr, atol=1e-5)}")
    print(f"v2 match: {torch.allclose(v2_t, v2_tr, atol=1e-5)}")


if __name__ == "__main__":
    profile()