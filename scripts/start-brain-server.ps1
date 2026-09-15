$ErrorActionPreference = "Stop"
$llama = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "C:\AI\llama.cpp\llama-server.exe" }
$model = if ($env:QWEN_BRAIN_MODEL) { $env:QWEN_BRAIN_MODEL } else { "C:\AI\models\Qwen3.8-27B-Uncensored-YMQ-XS-TI.gguf" }
$port = if ($env:QWEN_BRAIN_PORT) { $env:QWEN_BRAIN_PORT } else { "8080" }
$ctx = if ($env:QWEN_BRAIN_CTX) { $env:QWEN_BRAIN_CTX } else { "3072" }
$specType = if ($env:QWEN_BRAIN_SPEC_TYPE) { $env:QWEN_BRAIN_SPEC_TYPE } else { "draft-mtp" }
$specNMax = if ($env:QWEN_BRAIN_SPEC_DRAFT_N_MAX) { $env:QWEN_BRAIN_SPEC_DRAFT_N_MAX } else { "2" }
$kv = if ($env:QWEN_BRAIN_KV) { $env:QWEN_BRAIN_KV } else { "q4_0" }
$batch = if ($env:QWEN_BRAIN_BATCH) { $env:QWEN_BRAIN_BATCH } else { "256" }
$ubatch = if ($env:QWEN_BRAIN_UBATCH) { $env:QWEN_BRAIN_UBATCH } else { "128" }

Write-Host "brain llama-server  $model"
Write-Host "port                $port"
Write-Host "mtp                 $specType  n-max $specNMax  kv $kv  ctx $ctx  b $batch ub $ubatch"

$llamaArgs = @(
  "-m", $model,
  "-ngl", "99",
  "-np", "1",
  "--port", $port,
  "--host", "127.0.0.1",
  "-fa", "on",
  "--jinja",
  "--cache-prompt",
  "--load-mode", "none",
  "--no-webui",
  "--reasoning", "off",
  "-ctk", $kv,
  "-ctv", $kv,
  "-c", $ctx,
  "-b", $batch,
  "-ub", $ubatch
)
if ($specType -and ($specType -notmatch '^(?i)(none|off|0)$')) {
  $llamaArgs += @(
    "--spec-type", $specType,
    "--spec-draft-n-max", $specNMax,
    "--spec-draft-ngl", "99",
    "--spec-draft-type-k", $kv,
    "--spec-draft-type-v", $kv
  )
}

& $llama @llamaArgs
