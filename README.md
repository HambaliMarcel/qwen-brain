# Qwen brain

Local **Qwen3.5-4B** chat brain driven by live **Qwen3-ASR** speech-to-text. MVP: voice in, streamed text out, ultralow latency. ASR files are not modified; a new integrator module publishes the live stream.

```
mic → qwen3-asr-stream (port 9999, unchanged)
        └── integrator JSONL bus :18765
              └── qwen-brain listen
                    └── llama-server Qwen3.5-4B-Q8_0.gguf :8080
                          └── YOU / BRAIN text dashboard
```

Later: same 4B becomes the Hermes Agent core (`--backend hermes`), then TTS.

## Layout

| Piece | Path / port |
| --- | --- |
| This repo | `C:\Users\marce\Projects\qwen-brain` |
| ASR repo | `C:\Users\marce\Projects\qwen3-asr-stream` |
| Brain GGUF | `C:\AI\models\Qwen3.5-4B-Q8_0.gguf` |
| ASR GGUF | `C:\AI\models\Qwen3-ASR-1.7B-Q8_0.gguf` |
| llama-server | `C:\AI\llama.cpp\llama-server.exe` |
| ASR HTTP | `127.0.0.1:9999` |
| Brain HTTP | `127.0.0.1:8080` |
| STT bus | `127.0.0.1:18765` |
| Hermes API (future) | `127.0.0.1:8642` |

Two llama-server processes share the GPU: ASR 1.7B on 9999, chat 4B on 8080. Leave ASR running as you already do.

## Run (MVP)

One command starts ASR server, 4B server, live STT bus, and the brain dashboard. Extra windows open for the GPU servers and the mic; this window stays on **LIVE / YOU / BRAIN**. Already-running ports are reused.

```powershell
cd C:\Users\marce\Projects\qwen-brain
.\Start.bat
```

Or:

```powershell
.\scripts\start-all.ps1
.\scripts\start-all.ps1 --profile ultralow
```

Stop the stack:

```powershell
.\Stop.bat
```

Ctrl+C in the dashboard stops **listen** only (GPU servers stay warm). Use `Stop.bat` to kill everything.

Manual four-terminal commands are still in `scripts\` if you need them. Same mic flags as `python -m qwen3_asr_stream mic` (`--profile ultralow`, `--device`, …) pass through `start-all.ps1`.

Dashboard: **LIVE** = STT draft, **YOU** = command, **BRAIN** = streamed tokens. The window snaps to the left third of the screen like Cindy. STATUS shows mode, ttft, generation time, tok/s, e2e, prefill, token counts, and rolling averages. LOG stamps YOU / BRAIN / LAT / CUT / BUS / MODE.

Typed debug (no mic):

```powershell
python -m qwen_brain chat --start-server
```

## Latency

The brain does **not** wait for ASR LAST seal. It starts on:

1. **Eager pause** — `silence_sec >= 0.55` on a live lexical line (about a second before official commit).
2. **Official commit** — ASR LAST draft. Skipped if eager already sent the same line; restarted if LAST grew.

Thinking mode is forced off (`enable_thinking: false`). Replies are capped at 160 tokens. New speech cancels an in-flight reply (barge-in).

## Hermes (next, not MVP)

Point Hermes at this llama-server so the 4B is the agent core. In `C:\Users\marce\AppData\Local\hermes\config.yaml`:

```yaml
model:
  default: qwen3.5-4b
  provider: custom:local-llama
  base_url: http://127.0.0.1:8080/v1
  api_key: local
providers:
  local-llama:
    name: local llama.cpp
    api: http://127.0.0.1:8080/v1
    api_key: local
    default_model: qwen3.5-4b
```

Then:

```powershell
python -m qwen_brain listen --backend hermes
```

That posts committed speech to Hermes gateway (`http://127.0.0.1:8642/v1`) instead of calling the 4B directly, so tools and agentic workflows can run. TTS stays a stub (`qwen_brain/tts.py`) until a voice lane is wired.

## Tests

```powershell
cd C:\Users\marce\Projects\qwen-brain
python -m unittest discover -s tests -v
```

## Development branch

Active work lives on **`development`**. `main` is the stable pointer.
