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
| ASR GGUF | `C:\AI\models\Qwen3-ASR-1.7B-bf16.gguf` (mmproj stays Q8_0) |
| llama-server | `C:\AI\llama.cpp\llama-server.exe` |
| ASR HTTP | `127.0.0.1:9999` |
| Brain HTTP | `127.0.0.1:8080` |
| STT bus | `127.0.0.1:18765` |
| Hermes API (future) | `127.0.0.1:8642` |

Two llama-server processes share the GPU: ASR 1.7B on 9999, chat 4B on 8080. Leave ASR running as you already do.

## Run (MVP)

One command starts ASR server, 4B server, live STT bus, and the brain dashboard. Extra windows open for the GPU servers and the mic; this window stays on **YOU / BRAIN / LAT**. Already-running ports are reused.

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

Dashboard: sticky **STATUS** header, then Cindy-style finalized **YOU** (black settled transcript) / **BRAIN** (blue answer) turns, with the latest **LAT** pinned at the bottom. Draft ASR text and provisional 4B answers remain hidden. The normal Windows Terminal buffer and a scrolling log region preserve mouse-wheel and PageUp/PageDown history. PANN tags are attached to the next spoken YOU turn as scene context; a tag by itself never triggers an answer.

Typed debug (no mic):

```powershell
python -m qwen_brain chat --start-server
```

## Latency

The brain starts from **live STT**, using official Qwen streaming knobs:

1. **ASR hop** — adaptive `auto` starts at 600ms with official `unfixed_chunk_num=2` / `unfixed_token_num=5`, `max_tokens=32` per hop. This remains responsive while giving multilingual LID more audio than the English-heavy 400ms ultralow profile.
2. **Speculative live** — a draft that already looks complete starts the 4B after ~100ms of stable text, in parallel with the last ASR decode. This provisional work is not displayed or added to chat history.
3. **Eager pause** — `silence_sec`/`gap_sec` ≥ **0.12s**. Does **not** wait for ASR `decoding=false`.
4. **Live revision** — if the line grows by ~8+ characters, the HTTP stream is **aborted** so the GPU slot is freed, then the fuller line is generated provisionally.
5. **Settled LAST** — the bus waits for the full-utterance ASR refinement. If it matches the speculative draft, the ready answer is promoted instantly; if it changed, only the corrected final transcript is regenerated and shown.

Qwen3.5-4B sampling follows the non-thinking recipe (`temp 0.7`, `top_p 0.8`, `top_k 20`, `enable_thinking: false`, `reasoning_budget 0`). KV cache is Q8 to leave VRAM for bf16 ASR. Cancelled streams close the socket immediately so the dashboard cannot sit in THINKING behind a queued llama request.

PANN / ASR sound tags are `type=sound` on the bus. Replies cap at 120 tokens. A new utterance barges in.

The dashboard heartbeats while the bus is idle, resets THINKING after 6s / stalled answers after 8s, and does not flip back to WAITING on a bus keepalive.

Multilingual sessions default to Qwen3-ASR auto detection (`--language mix --no-lid-lock`). For a known monolingual session, an explicit model-supported language gives the strongest short-phrase accuracy without changing the default, for example `.\Start.bat --language Arabic` or `.\Start.bat --language Spanish`.

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
