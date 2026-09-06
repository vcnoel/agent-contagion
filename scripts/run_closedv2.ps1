# Corrected closed loop: injection at a compute step, so the corrupted call
# names no other account, and the placebo carries the canonical meaning.
# Waits for the 4B block to release the GPU, or gives up waiting after its
# log has been idle for 45 minutes.
$py = "$env:USERPROFILE\miniconda3\envs\gemma_spectral\python.exe"
$env:HF_HUB_OFFLINE = "1"
Set-Location "$PSScriptRoot/.."

Add-Type -Name PC -Namespace KC -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[KC.PC]::SetThreadExecutionState([uint32]2147483649) | Out-Null

while ($true) {
    $done = (Test-Path logs_q35_4b.txt) -and
            (Select-String -Path logs_q35_4b.txt -Pattern "Q35 4B DONE|4B UNAVAILABLE" -Quiet)
    $idle = (Test-Path logs_q35_4b.txt) -and
            (((Get-Date) - (Get-Item logs_q35_4b.txt).LastWriteTime).TotalMinutes -gt 45)
    if ($done -or $idle) { break }
    Start-Sleep 120
}
Start-Sleep 30

foreach ($m in @("qwen-3b", "qwen-1.5b", "qwen3.5-2b")) {
    Add-Content logs_closedv2.txt "=== closedv2 $m"
    & $py -m contagion.run_closed --model $m --n-tasks 30 `
        --out "results/closedv2_$m.jsonl" 2>&1 |
        Select-Object -Last 3 | Add-Content logs_closedv2.txt
}
Add-Content logs_closedv2.txt "CLOSEDV2 DONE"
[KC.PC]::SetThreadExecutionState([uint32]2147483648) | Out-Null
