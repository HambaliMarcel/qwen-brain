# One command: ASR server + 27B server + live STT bus + brain dashboard.
# Already-healthy ports are reused (no second GPU load).
# Extra args go to the ASR integrator (same flags as `python -m qwen3_asr_stream mic`).
#
#   .\scripts\start-all.ps1
#   .\scripts\start-all.ps1 --profile ultralow
#   .\Start.bat

param(
    [switch]$KillOnExit,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$IntegratorArgs
)

$ErrorActionPreference = "Stop"
try { chcp 65001 | Out-Null } catch {}
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$Host.UI.RawUI.WindowTitle = "brain  listen"

$BrainRoot = Split-Path -Parent $PSScriptRoot
$AsrRoot = if ($env:QWEN_ASR_ROOT) { $env:QWEN_ASR_ROOT } else { "C:\Users\marce\Projects\qwen3-asr-stream" }
$Llama = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "C:\AI\llama.cpp\llama-server.exe" }
$Models = if ($env:QWEN_ASR_MODELS_DIR) { $env:QWEN_ASR_MODELS_DIR } else { "C:\AI\models" }
# Q4_K_M: same VRAM class as Q4_0 but far less prone to decoder loops
# ("black on black on ...") on sung/music input. Q4_0 stays as fallback.
$AsrDefault = Join-Path $Models "Qwen3-ASR-1.7B-Q4_K_M.gguf"
if (-not (Test-Path -LiteralPath $AsrDefault)) { $AsrDefault = Join-Path $Models "Qwen3-ASR-1.7B-Q4_0.gguf" }
$AsrModel = if ($env:QWEN_ASR_MODEL) { $env:QWEN_ASR_MODEL } else { $AsrDefault }
if ([System.IO.Path]::GetFileName($AsrModel) -match '(?i)bf16') {
    $AsrModel = $AsrDefault
    Write-Host "ASR      bf16 is disabled; using $AsrModel"
}
if ([System.IO.Path]::GetFileName($AsrModel) -ceq "qwen3-asr-1.7b-q4_0.gguf") {
    Write-Host "ASR      cstr qwen3asr GGUF is CrispASR-only; using llama.cpp $AsrDefault"
    $AsrModel = $AsrDefault
}
$AsrMmproj = if ($env:QWEN_ASR_MMPROJ) { $env:QWEN_ASR_MMPROJ } else { Join-Path $Models "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" }
$AsrPort = if ($env:QWEN_ASR_PORT) { [int]$env:QWEN_ASR_PORT } else { 9999 }
$BrainModel = if ($env:QWEN_BRAIN_MODEL) { $env:QWEN_BRAIN_MODEL } else { "C:\AI\models\Qwen3.8-27B-Uncensored-YMQ-XS-TI.gguf" }
$BrainPort = if ($env:QWEN_BRAIN_PORT) { [int]$env:QWEN_BRAIN_PORT } else { 8080 }
$BrainCtx = if ($env:QWEN_BRAIN_CTX) { $env:QWEN_BRAIN_CTX } else { "3072" }
$BrainSpecType = if ($env:QWEN_BRAIN_SPEC_TYPE) { $env:QWEN_BRAIN_SPEC_TYPE } else { "draft-mtp" }
$BrainSpecDraftNMax = if ($env:QWEN_BRAIN_SPEC_DRAFT_N_MAX) { $env:QWEN_BRAIN_SPEC_DRAFT_N_MAX } else { "2" }
$BrainKv = if ($env:QWEN_BRAIN_KV) { $env:QWEN_BRAIN_KV } else { "q4_0" }
$BrainBatch = if ($env:QWEN_BRAIN_BATCH) { $env:QWEN_BRAIN_BATCH } else { "256" }
$BrainUbatch = if ($env:QWEN_BRAIN_UBATCH) { $env:QWEN_BRAIN_UBATCH } else { "128" }
$BusHost = if ($env:QWEN_BRAIN_STT_HOST) { $env:QWEN_BRAIN_STT_HOST } else { "127.0.0.1" }
$BusPort = if ($env:QWEN_BRAIN_STT_PORT) { [int]$env:QWEN_BRAIN_STT_PORT } else { 18765 }

$script:ChildPids = @()

function Test-HttpOk([string]$Url) {
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Test-PortOpen([string]$TargetHost, [int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $ok = $client.ConnectAsync($TargetHost, $Port).Wait(250)
        return [bool]$ok -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Wait-HttpOk([string]$Url, [int]$Seconds, [string]$Label) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    Write-Host "waiting  $Label  $Url"
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpOk $Url) {
            Write-Host "ready    $Label"
            return
        }
        Start-Sleep -Milliseconds 400
    }
    throw "Timed out waiting for $Label at $Url"
}

function Wait-PortOpen([string]$TargetHost, [int]$Port, [int]$Seconds, [string]$Label) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    Write-Host "waiting  $Label  ${TargetHost}:${Port}"
    while ((Get-Date) -lt $deadline) {
        if (Test-PortOpen $TargetHost $Port) {
            Write-Host "ready    $Label"
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Timed out waiting for $Label at ${TargetHost}:${Port}"
}

function ConvertTo-CmdArg([string]$Value) {
    if ($Value -match '[\s&<>^|()"]') {
        return '"' + ($Value -replace '"', '""') + '"'
    }
    return $Value
}

function Start-TitledProcess([string]$Title, [string]$WorkDir, [string]$File, [string[]]$Arguments) {
    $quotedFile = ConvertTo-CmdArg $File
    $quotedArgs = @($Arguments | ForEach-Object { ConvertTo-CmdArg $_ })
    $line = "title $Title && $quotedFile $($quotedArgs -join ' ')"
    $proc = Start-Process -FilePath "cmd.exe" -WorkingDirectory $WorkDir -PassThru -ArgumentList @("/k", $line)
    $script:ChildPids += $proc.Id
    Write-Host ("started  {0}  PID {1}" -f $Title, $proc.Id)
}

function Stop-Children {
    foreach ($childPid in $script:ChildPids) {
        try { Stop-Process -Id $childPid -Force -ErrorAction SilentlyContinue } catch {}
    }
}

if (-not (Test-Path -LiteralPath $Llama)) { throw "llama-server not found: $Llama" }
if (-not (Test-Path -LiteralPath $AsrRoot)) { throw "ASR repo not found: $AsrRoot" }
if (-not (Test-Path -LiteralPath $AsrModel)) { throw "ASR GGUF not found: $AsrModel" }
if (-not (Test-Path -LiteralPath $AsrMmproj)) { throw "ASR mmproj not found: $AsrMmproj" }
if (-not (Test-Path -LiteralPath $BrainModel)) { throw "brain GGUF not found: $BrainModel" }

Set-Location -LiteralPath $BrainRoot
$llamaDir = Split-Path -Parent $Llama

$flags = @()
if ($IntegratorArgs -and $IntegratorArgs.Count -gt 0) {
    $flags = @($IntegratorArgs)
}
if ($flags -notcontains "--profile") {
    $flags = @("--profile", "ultralow") + $flags
}
if ($flags -notcontains "--hop") {
    $flags += @("--hop", "0.50")
}
if ($flags -notcontains "--max-tokens") {
    $flags += @("--max-tokens", "32")
}
if ($flags -notcontains "--unfixed-chunks") {
    $flags += @("--unfixed-chunks", "2")
}
if ($flags -notcontains "--unfixed-tokens") {
    $flags += @("--unfixed-tokens", "5")
}
if ($flags -notcontains "--min-audio") {
    $flags += @("--min-audio", "0.40")
}
if ($flags -notcontains "--pann-interval") {
    $flags += @("--pann-interval", "0.8")
}
if ($flags -notcontains "--language") {
    $flags += @("--language", "mix")
}
if (($flags -notcontains "--lid-lock") -and ($flags -notcontains "--no-lid-lock")) {
    $flags += "--no-lid-lock"
}
if ($flags -notcontains "--silence-commit") {
    $flags += @("--silence-commit", "0.90")
}
if ($flags -notcontains "--silence-hangover") {
    $flags += @("--silence-hangover", "0.28")
}
if ($flags -notcontains "--no-auto-tune") {
    $flags += "--no-auto-tune"
}
if ($flags -notcontains "--no-refine") {
    $flags += "--no-refine"
}

Write-Host "Qwen stack  one-shot"
Write-Host "  ASR    $AsrModel  :$AsrPort"
Write-Host "  mmproj $AsrMmproj"
Write-Host "  brain  $BrainModel  :$BrainPort"
Write-Host "  mtp    $BrainSpecType  n-max $BrainSpecDraftNMax  kv $BrainKv  ctx $BrainCtx  b $BrainBatch ub $BrainUbatch"
Write-Host "  bus    ${BusHost}:${BusPort}"
Write-Host ""

$asrUrl = "http://127.0.0.1:$AsrPort/health"
$brainUrl = "http://127.0.0.1:$BrainPort/health"

if (Test-HttpOk $asrUrl) {
    $loaded = ""
    try {
        $props = Invoke-RestMethod -Uri "http://127.0.0.1:$AsrPort/props" -TimeoutSec 2
        $loaded = [string]($props.model_path)
        if (-not $loaded) { $loaded = [string]$props }
    } catch {}
    $want = [System.IO.Path]::GetFileName($AsrModel)
    if ($want -and $loaded -and ($loaded -notmatch [regex]::Escape($want))) {
        throw "port $AsrPort is running $loaded, want $want. Run Stop.bat then Start.bat."
    }
    Write-Host "reuse    ASR llama-server  :$AsrPort"
} else {
    Start-TitledProcess "Qwen ASR server" $llamaDir $Llama @(
        "-m", $AsrModel,
        "--mmproj", $AsrMmproj,
        "-ngl", "99",
        "-c", "4096",
        "-np", "1",
        "-n", "32",
        "--temp", "0.01",
        "--port", "$AsrPort",
        "--host", "127.0.0.1",
        "-fa", "on",
        "--jinja",
        "--prefill-assistant",
        "--cache-prompt",
        "--mmproj-offload",
        "--no-webui"
    )
    Wait-HttpOk $asrUrl 240 "ASR llama-server"
}

if (Test-HttpOk $brainUrl) {
    $loaded = ""
    $haveCtx = 0
    try {
        $props = Invoke-RestMethod -Uri "http://127.0.0.1:$BrainPort/props" -TimeoutSec 2
        $loaded = [string]($props.model_path)
        if (-not $loaded) { $loaded = [string]$props }
        $haveCtx = [int]($props.default_generation_settings.n_ctx)
    } catch {}
    $want = [System.IO.Path]::GetFileName($BrainModel)
    if ($want -and $loaded -and ($loaded -notmatch [regex]::Escape($want))) {
        throw "port $BrainPort is running $loaded, want $want. Run Stop.bat then Start.bat."
    }
    $wantCtx = [int]$BrainCtx
    if ($haveCtx -ne $wantCtx) {
        throw "port $BrainPort still has ctx $haveCtx, want $wantCtx. Run Stop.bat then Start.bat."
    }
    Write-Host "reuse    brain llama-server  :$BrainPort"
} else {
    $brainArgs = @(
        "-m", $BrainModel,
        "-ngl", "99",
        "-np", "1",
        "--port", "$BrainPort",
        "--host", "127.0.0.1",
        "-fa", "on",
        "--jinja",
        "--cache-prompt",
        "--load-mode", "none",
        "--no-webui",
        "--reasoning", "off",
        "-ctk", $BrainKv,
        "-ctv", $BrainKv,
        "-c", "$BrainCtx",
        "-b", "$BrainBatch",
        "-ub", "$BrainUbatch"
    )
    if ($BrainSpecType -and ($BrainSpecType -notmatch '^(?i)(none|off|0)$')) {
        $brainArgs += @(
            "--spec-type", $BrainSpecType,
            "--spec-draft-n-max", "$BrainSpecDraftNMax",
            "--spec-draft-ngl", "99",
            "--spec-draft-type-k", $BrainKv,
            "--spec-draft-type-v", $BrainKv
        )
    }
    Start-TitledProcess "Qwen brain server" $llamaDir $Llama $brainArgs
    Wait-HttpOk $brainUrl 240 "brain llama-server"
}

if (Test-PortOpen $BusHost $BusPort) {
    Write-Host "restart  STT bus  ${BusHost}:${BusPort}  (pick up hop/pause flags)"
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        $cmd = [string]$_.CommandLine
        if ($cmd -match "qwen3_asr_stream\.integrator") {
            Write-Host ("  stop integrator  PID {0}" -f $_.ProcessId)
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    $owned = Get-NetTCPConnection -LocalPort $BusPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in @($owned)) {
        if (-not $procId) { continue }
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($null -eq $p) { continue }
        if ($p.ProcessName -match "python") {
            Write-Host ("  stop port {0}  PID {1}" -f $BusPort, $procId)
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Milliseconds 400
}
$pyArgs = @(
    "-m", "qwen3_asr_stream.integrator",
    "--bus-host", $BusHost,
    "--bus-port", "$BusPort"
) + $flags
Start-TitledProcess "Qwen ASR bus" $AsrRoot "python" $pyArgs
Wait-PortOpen $BusHost $BusPort 60 "STT bus"

Write-Host ""
Write-Host "this window is the BRAIN dashboard.  Ctrl+C stops listen only."
Write-Host "Stop everything:  .\Stop.bat   or   .\scripts\stop-all.ps1"
Write-Host ""

try {
    python -m qwen_brain listen --stt-host $BusHost --stt-port $BusPort --port $BrainPort
} finally {
    if ($KillOnExit) {
        Write-Host "stopping windows started by this script..."
        Stop-Children
    }
}
