# Controls and reruns, queued behind the Llama measurement.
#
#  1. Matched escalation gave +1.00 on the Qwen3.5 pair when it protected
#     the submit step. Submit is also the SCORED action, so part of that
#     gain could be tautological rather than a vindication of the R0
#     ranking. Protecting read already gave +0.00 on the same pair; rate and
#     compute complete the control set. If only submit helps, the R0 ranking
#     predicts the right kind. If every non-read kind helps, it does not.
#  2. env4 reruns on all four models. The prompt now keeps strict
#     user/assistant alternation, which changed after the first three runs,
#     so the released code must reproduce the reported numbers.
#  3. Closed loop on Qwen3 4B. The intended Qwen3.5 4B has an incomplete
#     download (one shard of two), so a complete model stands in.
$py = "$env:USERPROFILE\miniconda3\envs\gemma_spectral\python.exe"
$env:HF_HUB_OFFLINE = "1"
Set-Location "$PSScriptRoot/.."

Add-Type -Name PC2 -Namespace KC2 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[KC2.PC2]::SetThreadExecutionState([uint32]2147483649) | Out-Null

while ($true) {
    $done = (Test-Path logs_llama8b.txt) -and
            (Select-String -Path logs_llama8b.txt -Pattern "LLAMA8B DONE" -Quiet)
    if ($done) { break }
    Start-Sleep 180
}
Start-Sleep 30

foreach ($k in @("rate", "compute")) {
    Add-Content logs_controls.txt "=== matched qwen3.5 pair, target $k"
    & $py -m contagion.run_free --small qwen3.5-0.8b --large qwen3.5-2b `
        --policies none,matched --target-kind $k --budgets 0.25 --n-tasks 30 `
        --out "results/matched_${k}_qwen3.5-0.8b_qwen3.5-2b.jsonl" 2>&1 |
        Select-Object -Last 2 | Add-Content logs_controls.txt
}

foreach ($m in @("qwen-1.5b", "qwen-3b", "qwen3.5-2b", "gemma-2b")) {
    Add-Content logs_env4b.txt "=== env4 rerun $m"
    & $py -m contagion.run_forced4 --model $m --n-tasks 30 --batch-size 16 `
        2>&1 | Select-Object -Last 2 | Add-Content logs_env4b.txt
}

Add-Content logs_closedv3.txt "=== closed qwen3-4b"
& $py -m contagion.run_closed --model qwen3-4b --n-tasks 30 `
    --out results/closedv2_qwen3-4b.jsonl 2>&1 |
    Select-Object -Last 3 | Add-Content logs_closedv3.txt

Add-Content logs_controls.txt "CONTROLS DONE"
[KC2.PC2]::SetThreadExecutionState([uint32]2147483648) | Out-Null
