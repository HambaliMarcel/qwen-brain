$ErrorActionPreference = "Stop"
$llama = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "C:\AI\llama.cpp\llama-server.exe" }
$model = if ($env:QWEN_BRAIN_MODEL) { $env:QWEN_BRAIN_MODEL } else { "C:\AI\models\Qwen3.8-27B-Uncensored-YMQ-XS-TI.gguf" }
$port = if ($env:QWEN_BRAIN_PORT) { $env:QWEN_BRAIN_PORT } else { "8080" }
$ctx = if ($env:QWEN_BRAIN_CTX) { $env:QWEN_BRAIN_CTX } else { "8192" }
$specType = if ($env:QWEN_BRAIN_SPEC_TYPE) { $env:QWEN_BRAIN_SPEC_TYPE } else { "draft-mtp" }
$specNMax = if ($env:QWEN_BRAIN_SPEC_DRAFT_N_MAX) { $env:QWEN_BRAIN_SPEC_DRAFT_N_MAX } else { "2" }

Write-Host "brain llama-server  $model"
Write-Host "port                $port"
Write-Host "mtp                 $specType  n-max $specNMax"

$llamaArgs = @(
  "-m", $model,
  "-ngl", "99",
  "-c", $ctx,
  "-np", "1",
  "--port", $port,
  "--host", "127.0.0.1",
  "-fa", "on",
  "--jinja",
  "--cache-prompt",
  "--no-webui",
  "--reasoning-budget", "0",
  "--chat-template-kwargs", '{"enable_thinking":false}',
  "-ctk", "q8_0",
  "-ctv", "q8_0"
)
if ($specType -and ($specType -notmatch '^(?i)(none|off|0)$')) {
  $llamaArgs += @(
    "--spec-type", $specType,
    "--spec-draft-n-max", $specNMax,
    "--spec-draft-ngl", "99"
  )
}

& $llama @llamaArgs
