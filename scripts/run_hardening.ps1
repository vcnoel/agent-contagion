# Hardening runs. The GPU is free: the Llama download finished and its
# measurement is queued last, behind everything here.
#
#  1. env4, free text: no CALL template, answers in ordinary sentences. Two
#     turn kinds in one task, one governed by a convention the model must
#     induce and one a plain lookup, so the mechanism claim is tested as a
#     within task dissociation.
#  2. Matched escalation: protect the step kind each driver's own R0
#     decomposition names, rather than assuming the dispatch.
#  3. Closed loop on a second capable driver.
$py = "$env:USERPROFILE\miniconda3\envs\gemma_spectral\python.exe"
$env:HF_HUB_OFFLINE = "1"
Set-Location "$PSScriptRoot/.."

Add-Type -Name PH2 -Namespace KH2 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[KH2.PH2]::SetThreadExecutionState([uint32]2147483649) | Out-Null

foreach ($m in @("qwen-1.5b", "qwen-3b", "qwen3.5-2b", "gemma-2b")) {
    Add-Content logs_env4.txt "=== env4 $m"
    & $py -m contagion.run_forced4 --model $m --n-tasks 30 --batch-size 16 `
        2>&1 | Select-Object -Last 2 | Add-Content logs_env4.txt
}

# Qwen2.5's superspreader is the dispatching read, Qwen3.5's the submission.
Add-Content logs_matched.txt "=== matched qwen-0.5b -> qwen-1.5b (read)"
& $py -m contagion.run_free --small qwen-0.5b --large qwen-1.5b `
    --policies none,matched --target-kind read --budgets 0.25 --n-tasks 30 `
    --out results/matched_qwen-0.5b_qwen-1.5b.jsonl 2>&1 |
    Select-Object -Last 2 | Add-Content logs_matched.txt

Add-Content logs_matched.txt "=== matched qwen3.5-0.8b -> qwen3.5-2b (submit)"
& $py -m contagion.run_free --small qwen3.5-0.8b --large qwen3.5-2b `
    --policies none,matched --target-kind submit --budgets 0.25 --n-tasks 30 `
    --out results/matched_qwen3.5-0.8b_qwen3.5-2b.jsonl 2>&1 |
    Select-Object -Last 2 | Add-Content logs_matched.txt

Add-Content logs_closedv3.txt "=== closed qwen3.5-4b"
& $py -m contagion.run_closed --model qwen3.5-4b --n-tasks 30 `
    --out results/closedv2_qwen3.5-4b.jsonl 2>&1 |
    Select-Object -Last 3 | Add-Content logs_closedv3.txt

Add-Content logs_hardening.txt "HARDENING DONE"
[KH2.PH2]::SetThreadExecutionState([uint32]2147483648) | Out-Null
