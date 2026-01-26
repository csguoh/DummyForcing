import torch
import triton
import triton.language as tl
import time


@triton.jit
def rope_apply_kernel(
        x_ptr, freqs0_ptr, freqs1_ptr, freqs2_ptr, output_ptr,
        start_frame,
        B, L, num_heads, C,
        stride_xb, stride_xl, stride_xh, stride_xc,
        stride_ob, stride_ol, stride_oh, stride_oc,
        c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr, h: tl.constexpr, w: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_b >= B or pid_l >= L or pid_h >= num_heads:
        return

    frame_idx = pid_l // (h * w)
    hw_idx = pid_l % (h * w)
    h_idx = hw_idx // w
    w_idx = hw_idx % w

    c_half = C // 2

    for c_start in range(0, c_half, BLOCK_SIZE):
        offsets = c_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < c_half

        x_offset = pid_b * stride_xb + pid_l * stride_xl + pid_h * stride_xh + offsets * 2
        x_real = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        x_imag = tl.load(x_ptr + x_offset + 1, mask=mask, other=0.0)

        freq_real = tl.zeros_like(x_real)
        freq_imag = tl.zeros_like(x_imag)

        in_c0 = offsets < c0
        in_c1 = (offsets >= c0) & (offsets < c0 + c1)
        in_c2 = offsets >= c0 + c1

        freq_idx_0 = (start_frame + frame_idx) * c0 * 2 + (offsets - 0) * 2
        freq_idx_1 = h_idx * c1 * 2 + (offsets - c0) * 2
        freq_idx_2 = w_idx * c2 * 2 + (offsets - c0 - c1) * 2

        f_real_0 = tl.load(freqs0_ptr + freq_idx_0, mask=in_c0, other=1.0)
        f_imag_0 = tl.load(freqs0_ptr + freq_idx_0 + 1, mask=in_c0, other=0.0)

        f_real_1 = tl.load(freqs1_ptr + freq_idx_1, mask=in_c1, other=1.0)
        f_imag_1 = tl.load(freqs1_ptr + freq_idx_1 + 1, mask=in_c1, other=0.0)

        f_real_2 = tl.load(freqs2_ptr + freq_idx_2, mask=in_c2, other=1.0)
        f_imag_2 = tl.load(freqs2_ptr + freq_idx_2 + 1, mask=in_c2, other=0.0)

        freq_real = tl.where(in_c0, f_real_0, freq_real)
        freq_imag = tl.where(in_c0, f_imag_0, freq_imag)
        freq_real = tl.where(in_c1, f_real_1, freq_real)
        freq_imag = tl.where(in_c1, f_imag_1, freq_imag)
        freq_real = tl.where(in_c2, f_real_2, freq_real)
        freq_imag = tl.where(in_c2, f_imag_2, freq_imag)

        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real

        out_offset = pid_b * stride_ob + pid_l * stride_ol + pid_h * stride_oh + offsets * 2
        tl.store(output_ptr + out_offset, out_real, mask=mask)
        tl.store(output_ptr + out_offset + 1, out_imag, mask=mask)


def rope_apply_triton(x, grid_size, freqs, start_frame=0):
    B, L, num_heads, C = x.shape
    output = torch.empty_like(x)

    c_half = C // 2
    c0 = c_half - 2 * (c_half // 3)
    c1 = c_half // 3
    c2 = c_half // 3

    freqs_split = freqs.split([c0, c1, c2], dim=1)
    freqs0 = torch.view_as_real(freqs_split[0]).reshape(-1)
    freqs1 = torch.view_as_real(freqs_split[1]).reshape(-1)
    freqs2 = torch.view_as_real(freqs_split[2]).reshape(-1)

    grid = (B, L, num_heads)
    BLOCK_SIZE = 64
    h, w = grid_size[0][1].item(), grid_size[0][2].item()
    rope_apply_kernel[grid](
        x, freqs0, freqs1, freqs2, output,
        start_frame,
        B, L, num_heads, C,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        c0=c0, c1=c1, c2=c2, h=h, w=w,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).type_as(x)


def profile():
    device = 'cuda'
    B, L, num_heads, C = 1, 4680, 12, 128

    x = torch.randn(B, L, num_heads, C, device=device, dtype=torch.float32)
    freqs = torch.randn(1024, 64, device=device, dtype=torch.complex64)
    grid_sizes = torch.tensor([[3, 30, 52]], device=device)
    start_frame = 0

    # Warmup
    for _ in range(10):
        _ = causal_rope_apply(x, grid_sizes, freqs, start_frame)
        _ = rope_apply_triton(x, freqs, start_frame)

    torch.cuda.synchronize()

    # Benchmark original
    num_runs = 100
    start = time.perf_counter()
    torch.cuda.synchronize()

    for _ in range(num_runs):
        out_orig = causal_rope_apply(x, grid_sizes, freqs, start_frame)

    torch.cuda.synchronize()
    orig_time = (time.perf_counter() - start) * 1000 / num_runs

    # Benchmark triton
    start = time.perf_counter()
    torch.cuda.synchronize()

    for _ in range(num_runs):
        out_triton = rope_apply_triton(x, freqs, start_frame)

    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) * 1000 / num_runs

    # Verify correctness
    out_orig = causal_rope_apply(x, grid_sizes, freqs, start_frame)
    out_triton = rope_apply_triton(x, freqs, start_frame)

    print(f"Original time: {orig_time:.3f} ms")
    print(f"Triton time: {triton_time:.3f} ms")
    print(f"Speedup: {orig_time / triton_time:.2f}x")
    print(f"\nCorrectness check:")
    print(f"Max diff: {torch.max(torch.abs(out_orig - out_triton)).item():.6f}")
    print(f"Allclose (atol=1e-4): {torch.allclose(out_orig, out_triton, atol=1e-4)}")


if __name__ == "__main__":
    profile()