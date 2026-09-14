# Stop the one-shot stack: STT bus, listen, and llama-server on ASR/brain ports.
# GPU servers are killed only if they own 9999 / 8080.

param(
    [int[]]$Ports = @(9999, 8080, 18765)
)

$ErrorActionPreference = "SilentlyContinue"
Write-Host "stopping Qwen stack..."

Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" | ForEach-Object {
    $cmd = [string]$_.CommandLine
    if ($cmd -match "qwen3_asr_stream\.integrator|qwen_brain listen|qwen_brain serve") {
        Write-Host ("  python  PID {0}" -f $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force
    }
}

Get-Process -Name powershell -ErrorAction SilentlyContinue | Where-Object {
    $_.MainWindowTitle -match "^Qwen (ASR server|brain server|ASR bus|brain  listen)"
} | ForEach-Object {
    Write-Host ("  window  {0}  PID {1}" -f $_.MainWindowTitle, $_.Id)
    Stop-Process -Id $_.Id -Force
}

foreach ($port in $Ports) {
    $owned = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $owned) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($null -eq $p) { continue }
        Write-Host ("  port {0}  {1}  PID {2}" -f $port, $p.ProcessName, $procId)
        Stop-Process -Id $procId -Force
    }
}

Write-Host "done"
