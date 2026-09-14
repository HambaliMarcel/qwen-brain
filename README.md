# Qwen brain

Local **Qwen3.8-27B** chat brain driven by live **Qwen3-ASR** speech-to-text. MVP: voice in, streamed text out, ultralow latency. ASR files are not modified; a new integrator module publishes the live stream.

```
mic → qwen3-asr-stream (port 9999, unchanged)
        └── integrator JSONL bus :18765
              └── qwen-brain listen
                    └── llama-server Qwen3.8-27B-Uncensored-YMQ-XS-TI.gguf :8080
                          └── YOU / BRAIN text dashboard
```

Later: same 27B becomes the Hermes Agent core (`--backend hermes`), then TTS.

## Layout

| Piece | Path / port |
| --- | --- |
| This repo | `C:\Users\marce\Projects\qwen-brain` |
| ASR repo | `C:\Users\marce\Projects\qwen3-asr-stream` |
| Brain GGUF | `C:\AI\models\Qwen3.8-27B-Uncensored-YMQ-XS-TI.gguf` (MTP `--spec-type draft-mtp`) |
| ASR GGUF | `C:\AI\models\Qwen3-ASR-1.7B-Q8_0.gguf` (mmproj `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`) |
| llama-server | `C:\AI\llama.cpp\llama-server.exe` |
| ASR HTTP | `127.0.0.1:9999` |
| Brain HTTP | `127.0.0.1:8080` |
| STT bus | `127.0.0.1:18765` |
| Hermes API (future) | `127.0.0.1:8642` |

Two llama-server processes share the GPU: ASR 1.7B Q8_0 on 9999, chat 27B on 8080 with MTP. Leave ASR running as you already do.

## Run (MVP)

One command starts ASR server, 27B server, live STT bus, and the brain dashboard. Extra windows open for the GPU servers and the mic; this window stays on **YOU / BRAIN / LAT**. Already-running ports are reused.

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

Dashboard: sticky **STATUS** header, then Cindy-style finalized **YOU** / **BRAIN** turns only, with **LAT** pinned at the bottom. Live drafts stay hidden. Room noise (`[batuk?]`, `[crowing]`, `[typing]`) stays on STATUS and does not start a 27B turn. Spoken words finalize when the utterance is actually finished: a complete line after a short silence, or an incomplete line after a real hold / LAST — never on an 80 ms syllable gap. The 27B answers calmly from the last five turns, without repeating a question you already answered.

Typed debug (no mic):

```powershell
python -m qwen_brain chat --start-server
```

## Latency

The brain starts from **live STT**, using official Qwen3-ASR mix streaming:

1. **ASR hop** — default **500ms** (`--no-auto-tune`), matching the official streaming example's smallest step. Mix mode uses `language=None` (no `force_language`). Continuation prefills the **model's own** `language X` tag, not an Indonesian/English lexicon guess.
2. **First decode** — `--min-audio 0.40` so the first hop has enough audio for LID.
3. **Endpoint** — a finished sentence may send after ~450ms of **silence**. Incomplete drafts wait until ~900ms of silence or LAST. Syllable `gap_sec` is ignored. Hidden speculation only runs while you are still talking a complete line.
4. **Shared GPU** — `--no-refine` so ASR does not start a second full decode that blocks the 27B. Replies stay short (`max_tokens` 48). Brain llama-server uses `--spec-type draft-mtp --spec-draft-n-max 2` (this GGUF has `nextn_predict_layers=1`).
5. **LAST** — still upgrades a truncated eager line; it is not required if the utterance already ended cleanly.

Qwen3.8-27B sampling stays short for voice (`temp 0.55`, `top_p 0.85`, `top_k 20`, `enable_thinking: false`, `reasoning_budget 0`). History keeps the last **5** user/assistant turns. KV cache is Q8 so ASR Q8_0 and the 27B fit on one GPU. Cancelled streams close the socket immediately so the dashboard cannot sit in THINKING behind a queued llama request.

PANN / ASR sound tags are `type=sound` for STATUS only. They are not committed as YOU lines. A new spoken utterance barges in.

The dashboard heartbeats while the bus is idle, resets THINKING after 6s / stalled answers after 8s, and does not flip back to WAITING on a bus keepalive.

Multilingual sessions default to Qwen3-ASR automatic LID (`--language mix --no-lid-lock`). Do not force Indonesian or English unless the session is truly monolingual. For a known monolingual session: `.\Start.bat --language Arabic` or `.\Start.bat --language Cantonese`.

## Hermes (next, not MVP)

Point Hermes at this llama-server so the 27B is the agent core. In `C:\Users\marce\AppData\Local\hermes\config.yaml`:

```yaml
model:
  default: qwen3.8-27b
  provider: custom:local-llama
  base_url: http://127.0.0.1:8080/v1
  api_key: local
providers:
  local-llama:
    name: local llama.cpp
    api: http://127.0.0.1:8080/v1
    api_key: local
    default_model: qwen3.8-27b
```

Then:

```powershell
python -m qwen_brain listen --backend hermes
```

That posts committed speech to Hermes gateway (`http://127.0.0.1:8642/v1`) instead of calling the 27B directly, so tools and agentic workflows can run. TTS stays a stub (`qwen_brain/tts.py`) until a voice lane is wired.

## Tests

```powershell
cd C:\Users\marce\Projects\qwen-brain
python -m unittest discover -s tests -v
```

## Development branch

Active work lives on **`development`**. `main` is the stable pointer.
