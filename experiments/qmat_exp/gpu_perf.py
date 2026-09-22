from __future__ import annotations

import statistics

from .io import read_json, resolve_path
from .native_data import read_native_data


def _time_cuda(fn, torch, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _quantize_symmetric(x, bits: int, torch, columnwise=False):
    q = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(dim=0, keepdim=True) if columnwise else x.abs().amax()).clamp_min(1e-30) / q
    ints = torch.clamp(torch.round(x / scale), -q, q).to(torch.int8)
    return ints, scale


def _row(bits: int, kernel: str, times: list[float], baseline: float, device: str, m: int, n: int, rank: int) -> dict:
    median = statistics.median(times)
    return {
        "device": device,
        "m": m,
        "n": n,
        "rank": rank,
        "bits": bits,
        "kernel": kernel,
        "median_ms": median,
        "min_ms": min(times),
        "max_ms": max(times),
        "speedup_vs_fp32": baseline / median if median > 0.0 else "",
    }


def run_gpu_performance(cfg: dict) -> list[dict]:
    import torch

    gpu_cfg = cfg["gpu"]
    if any(bit not in (4, 8) for bit in gpu_cfg["bits"]):
        raise ValueError("PyTorch proxy supports only 4/8 bits; use run_native.py for 12/16 bits")
    if not torch.cuda.is_available():
        return []
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    m = int(gpu_cfg["m"])
    n = int(gpu_cfg["n"])
    warmup = int(gpu_cfg["warmup"])
    repeats = int(gpu_cfg["repeats"])
    rows: list[dict] = []
    manifest = read_json(resolve_path(cfg["data_dir"]) / "manifest.json")
    cases = {item["rank"]: item for item in manifest.get("native_gpu", [])
             if item["m"] == m and item["n"] == n and not item.get("source_case")}

    for rank in gpu_cfg["ranks"]:
        if rank not in cases:
            raise ValueError("Missing saved GPU inputs; rerun prepare_data.py with matching native_shapes")
        _, p_host, q_host = read_native_data(resolve_path(cases[rank]["path"]))
        p = torch.from_numpy(p_host).to(device)
        q = torch.from_numpy(q_host).to(device)
        out_holder = {}

        def fp32_gemm():
            out_holder["fp32"] = p @ q.T

        fp32_times = _time_cuda(fp32_gemm, torch, warmup, repeats)
        fp32_median = statistics.median(fp32_times)
        rows.append(_row(32, "fp32_gemm", fp32_times, fp32_median, device_name, m, n, rank))

        for bits in gpu_cfg["bits"]:
            p_int, p_scale = _quantize_symmetric(p, bits, torch, columnwise=True)
            q_int, q_scale = _quantize_symmetric(q, bits, torch, columnwise=True)

            def columnwise_dequant_gemm():
                pd = p_int.float() * p_scale
                qd = q_int.float() * q_scale
                out_holder["dequant"] = pd @ qd.T

            dequant_times = _time_cuda(columnwise_dequant_gemm, torch, warmup, repeats)
            rows.append(_row(bits, "columnwise_dequant_gemm", dequant_times, fp32_median, device_name, m, n, rank))

            if bits == 8 and hasattr(torch, "_int_mm"):
                ps_int, ps_scale = _quantize_symmetric(p, bits, torch)
                qs_int, qs_scale = _quantize_symmetric(q, bits, torch)
                def shared_scale_int8_gemm():
                    acc = torch._int_mm(ps_int, qs_int.T.contiguous())
                    out_holder["int8"] = acc.float() * (ps_scale * qs_scale)

                int8_times = _time_cuda(shared_scale_int8_gemm, torch, warmup, repeats)
                rows.append(_row(bits, "shared_scale_int8_gemm", int8_times, fp32_median, device_name, m, n, rank))

    return rows
