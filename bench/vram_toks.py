"""Throwaway ASR+brain VRAM / tok-s bench. Does not bind production 9999/8080."""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

LLAMA = Path(r"C:\AI\llama.cpp\llama-server.exe")
BRAIN = Path(r"C:\AI\models\Qwen3.8-27B-Uncensored-YMQ-XS-TI.gguf")
ASR = Path(r"C:\AI\models\Qwen3-ASR-1.7B-Q4_K_M.gguf")
MMPROJ = Path(r"C:\AI\models\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
ASR_PORT = 19999
BRAIN_PORT = 18080
LOG_DIR = Path(__file__).resolve().parent / "logs"


def ps_dedicated() -> dict[int, int]:
    cmd = (
        "Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' | "
        "Select-Object -ExpandProperty CounterSamples | "
        "ForEach-Object { '{0}|{1}' -f $_.InstanceName, $_.CookedValue }"
    )
    out = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", cmd],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    pid_mb: dict[int, int] = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        name, raw = line.split("|", 1)
        m = re.search(r"pid_(\d+)", name, re.I)
        if not m:
            continue
        mb = int(round(float(raw) / (1024 * 1024)))
        if mb <= 0:
            continue
        pid = int(m.group(1))
        pid_mb[pid] = max(mb, pid_mb.get(pid, 0))
    return pid_mb


def llama_mb(pids: list[int]) -> tuple[int, int, dict[int, int]]:
    dedicated = ps_dedicated()
    per = {pid: dedicated.get(pid, 0) for pid in pids if pid}
    llama = sum(per.values())
    total = sum(dedicated.values())
    return llama, total, per


def wait_http(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last = "not ready"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return
        except Exception as exc:
            last = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"{url} {last}")


def start_server(args: list[str], log_name: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = (LOG_DIR / log_name).open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(LLAMA), *args],
        cwd=str(LLAMA.parent),
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    proc._log = handle  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    handle = getattr(proc, "_log", None)
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    if handle:
        try:
            handle.close()
        except Exception:
            pass


def chat(max_tokens: int = 32) -> dict:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are Marcelino's close friend. One or two short spoken lines.",
            },
            {"role": "user", "content": "I'm in love with you"},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "max_tokens": max_tokens,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{BRAIN_PORT}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    text = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return {
        "wall_ms": round(wall_ms, 1),
        "prompt_ms": round(float(timings.get("prompt_ms") or 0), 1),
        "pred_ms": round(float(timings.get("predicted_ms") or 0), 1),
        "prompt_n": int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0),
        "pred_n": int(timings.get("predicted_n") or usage.get("completion_tokens") or 0),
        "tok_s": round(float(timings.get("predicted_per_second") or 0), 2),
        "text": text[:80],
    }


def asr_args(cfg: dict) -> list[str]:
    ngl = str(cfg.get("asr_ngl", 99))
    args = [
        "-m", str(ASR),
        "--mmproj", str(MMPROJ),
        "-ngl", ngl,
        "-c", "1536",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-b", "512",
        "-ub", "256",
        "-np", "1",
        "-n", "32",
        "--temp", "0.01",
        "--port", str(ASR_PORT),
        "--host", "127.0.0.1",
        "-fa", "on",
        "--jinja",
        "--prefill-assistant",
        "--cache-prompt",
        "--cache-ram", "0",
        "--load-mode", "none",
        "--no-webui",
    ]
    if cfg.get("asr_mmproj_offload", True) and ngl != "0":
        args.append("--mmproj-offload")
    return args


def brain_args(cfg: dict) -> list[str]:
    args = [
        "-m", str(BRAIN),
        "-np", "1",
        "--port", str(BRAIN_PORT),
        "--host", "127.0.0.1",
        "-fa", "on",
        "--jinja",
        "--cache-prompt",
        "--load-mode", "none",
        "--no-webui",
        "--reasoning", "off",
        "-ctk", cfg["kv"],
        "-ctv", cfg["kv"],
    ]
    if cfg.get("ngl") is not None:
        args.extend(["-ngl", str(cfg["ngl"])])
    if cfg.get("ctx"):
        args.extend(["-c", str(cfg["ctx"])])
    if cfg.get("fit"):
        args.extend(
            [
                "--fit", "on",
                "--fit-target", str(cfg["fit"]),
                "--fit-ctx", str(cfg.get("fit_ctx", 2048)),
            ]
        )
    if cfg.get("mtp"):
        draft = [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(cfg.get("mtp_n", 2)),
            "--spec-draft-type-k", cfg["kv"],
            "--spec-draft-type-v", cfg["kv"],
        ]
        if cfg.get("ngl") is not None:
            draft.extend(["--spec-draft-ngl", str(cfg["ngl"])])
        args.extend(draft)
    if cfg.get("ub"):
        args.extend(["-ub", str(cfg["ub"])])
    if cfg.get("b"):
        args.extend(["-b", str(cfg["b"])])
    return args


CONFIGS = [
    {"id": "q4_c3072_mtp", "kv": "q4_0", "ctx": 3072, "ngl": 99, "fit": None, "mtp": True},
    {"id": "q4_c3072_mtp_ub128", "kv": "q4_0", "ctx": 3072, "ngl": 99, "fit": None, "mtp": True, "b": 256, "ub": 128},
    {"id": "q4_c3072_mtp_asr_cpu", "kv": "q4_0", "ctx": 3072, "ngl": 99, "fit": None, "mtp": True, "asr_ngl": 0, "asr_mmproj_offload": False},
    {"id": "q4_c4096_mtp", "kv": "q4_0", "ctx": 4096, "ngl": 99, "fit": None, "mtp": True},
]


def run_one(cfg: dict) -> dict:
    asr = brain = None
    try:
        asr = start_server(asr_args(cfg), f"asr-{cfg['id']}.log")
        wait_http(f"http://127.0.0.1:{ASR_PORT}/health", 180)
        brain = start_server(brain_args(cfg), f"brain-{cfg['id']}.log")
        wait_http(f"http://127.0.0.1:{BRAIN_PORT}/health", 240)
        time.sleep(1.0)
        warmup = chat(16)
        runs = [chat(32) for _ in range(3)]
        llama_vram, total_vram, per = llama_mb([asr.pid, brain.pid])
        tok = [r["tok_s"] for r in runs if r["tok_s"] > 0]
        wall = [r["wall_ms"] for r in runs]
        prompt = [r["prompt_ms"] for r in runs]
        return {
            "id": cfg["id"],
            "ok": True,
            "vram_mb": llama_vram,
            "asr_vram_mb": per.get(asr.pid, 0),
            "brain_vram_mb": per.get(brain.pid, 0),
            "total_dedicated_mb": total_vram,
            "asr_pid": asr.pid,
            "brain_pid": brain.pid,
            "avg_tok_s": round(sum(tok) / len(tok), 2) if tok else 0.0,
            "avg_wall_ms": round(sum(wall) / len(wall), 1),
            "avg_prompt_ms": round(sum(prompt) / len(prompt), 1),
            "warmup_tok_s": warmup.get("tok_s"),
            "runs": runs,
        }
    except Exception as exc:
        return {"id": cfg["id"], "ok": False, "error": str(exc)}
    finally:
        stop_server(brain)
        stop_server(asr)
        time.sleep(2.0)


def main() -> int:
    rows = []
    for cfg in CONFIGS:
        print(f"\n== {cfg['id']}", flush=True)
        row = run_one(cfg)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "runs"}, ensure_ascii=False), flush=True)
    out = LOG_DIR / "vram_toks.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(
        f"{'id':24s} {'ok':5s} {'pair':>7s} {'asr':>6s} {'brain':>7s} {'all':>7s} "
        f"{'tok/s':>7s} {'wall':>8s} {'prefill':>8s}"
    )
    for row in rows:
        if not row.get("ok"):
            print(f"{row['id']:24s} FAIL  {row.get('error','')[:80]}")
            continue
        print(
            f"{row['id']:24s} ok    {row['vram_mb']:5d}MB {row.get('asr_vram_mb',0):5d} "
            f"{row.get('brain_vram_mb',0):5d} {row.get('total_dedicated_mb',0):5d}MB "
            f"{row['avg_tok_s']:7.2f} {row['avg_wall_ms']:7.0f}ms {row['avg_prompt_ms']:7.0f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
