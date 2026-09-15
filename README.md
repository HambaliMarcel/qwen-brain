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
| ASR GGUF | `C:\AI\models\Qwen3-ASR-1.7B-Q4_K_M.gguf` (mmproj `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`; Q4_0 is the fallback) |
| llama-server | `C:\AI\llama.cpp\llama-server.exe` |
| ASR HTTP | `127.0.0.1:9999` |
| Brain HTTP | `127.0.0.1:8080` |
| STT bus | `127.0.0.1:18765` |
| Hermes API (future) | `127.0.0.1:8642` |

Two llama-server processes share the GPU: ASR 1.7B Q4_K_M on 9999, chat 27B on 8080 with MTP. Q4_K_M replaced Q4_0 because the legacy Q4_0 decoder spiralled into repeats ("black on black on …") on sung/music input; the stream also has a loop guard (see Latency). Leave ASR running as you already do. The Hugging Face `cstr/qwen3-asr-1.7b-GGUF` file is a CrispASR `qwen3asr` pack; this llama-server needs the split `qwen3vl` decoder + mmproj instead.

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
3. **Endpoint** — a finished sentence may send after ~450ms of **silence**. Incomplete drafts and LAST wait ~1.5s so a sung or slow line is not chopped on every breath. One- to three-word LAST pieces are held and stitched before the 27B starts, so the GPU stays free for ASR.
4. **Shared GPU** — `--no-refine` so ASR does not start a second full decode that blocks the 27B. Casual replies are one spoken line (budget 48 tokens, 24 for a one/two-word turn); a detailed or long prompt raises the token budget to 384. Brain llama-server keeps all 65 layers on GPU (`-ngl 99`), Q4 KV, ctx 3072, MTP (`--spec-draft-n-max 2`), `-b 256 -ub 128`, `--load-mode none`. Do not pair `--fit` with `-ngl 99` (llama.cpp aborts the fit and can inflate ctx). ASR runs `-c 1536`, q8 KV, `-b 512 -ub 256`, `--cache-ram 0`. The two llama processes sit near **12.2 GB** (27B ≈ 10.0, ASR ≈ 2.3); putting 27B layers on CPU to force 11–12 GB drops tok/s from ~28 to ~2–4.
5. **Loop guard** — when the ASR decoder spirals ("black on black on …", "baby, baby, baby, …"), the stream aborts that decode as soon as the repeat shows up in the partial text, keeps only the words before the loop, drops the window's audio and sends DRY sampling for the next ~8 s. The brain also refuses a degenerate line, so a loop never becomes a 300-token prompt. A sung hook repeated 2–3 times is left alone.
6. **Turn dedupe** — LAST re-sends the grown paragraph; the brain strips what it already answered and sends only the new tail, so it does not answer the same lyric twice and the prompt stays short.
7. **LAST** — still upgrades a truncated eager line; it is not required if the utterance already ended cleanly.

Qwen3.8-27B sampling (`temp 0.7`, `top_p 0.9`, `top_k 20`, `--reasoning off`) stays friendly and a bit toxic, but it obeys the ask. History keeps the last **5** user/assistant turns and is trimmed **two turns at a time**: dropping one turn per ask shifts every later message and llama-server re-prefills the whole history (TTFT ~1.1 s at the cap); block trimming keeps the KV prefix stable in between (~0.7 s avg, ~0.5 s on a cache hit). The session also warms the system-prompt KV at startup so the first turn is not a 1.5 s cold start. Measured with ASR co-resident: casual turns e2e ~0.7–1.5 s at ~20–26 tok/s. Cancelled streams close the socket immediately so the dashboard cannot sit in THINKING behind a queued llama request.

**VRAM is the real "stuck" switch.** Other apps on this PC already hold ~3.5 GB of dedicated VRAM (dwm alone ~2.2 GB, then Edge, Cursor, Spotify). 12.2 + 3.5 GB is right at the 16 GB line; when it tips over, the driver pages VRAM over PCIe and the 27B drops to ~8 tok/s with 7–10 s TTFT until it settles. If that happens, close GPU-heavy tabs/apps first. As a last resort set `QWEN_ASR_MMPROJ_CPU=1` before `Start.bat`: the audio encoder moves to CPU (−900 MB VRAM) but each ASR hop goes from ~130 ms to ~500 ms.

**SSD spikes.** Both servers now use `--load-mode none` (no mmap). With mmap, the 10 GB weight file is file-backed and Windows keeps only ~20 MB of it cached, so CPU-side tensor reads re-fault from disk mid-session (2.8 M page faults vs 0.9 M). Host working set is ~180 MB either way; tok/s is unchanged.

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
