# Qwen3.5 block. Each model is load verified before any GPU time is spent on
# it, so an incomplete download is skipped rather than crashing mid queue.
$py = "$env:USERPROFILE\miniconda3\envs\gemma_spectral\python.exe"
$env:HF_HUB_OFFLINE = "1"          # never start a multi-GB fetch mid run
Set-Location "$PSScriptRoot/.."

Add-Type -Name PW -Namespace KQ -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[KQ.PW]::SetThreadExecutionState([uint32]2147483649) | Out-Null

$models = @(
  @{ tag = "qwen3.5-0.8b"; repo = "Qwen/Qwen3.5-0.8B"; bs = 24 },
  @{ tag = "qwen3.5-2b";   repo = "Qwen/Qwen3.5-2B";   bs = 16 },
  @{ tag = "qwen3.5-4b";   repo = "Qwen/Qwen3.5-4B";   bs = 12 }
)

$ok = @()
foreach ($m in $models) {
    & $py -c "from transformers import AutoModelForCausalLM; import torch; AutoModelForCausalLM.from_pretrained('$($m.repo)', dtype=torch.bfloat16, device_map='cpu'); print('VERIFIED')" 2>&1 |
        Select-String "VERIFIED" | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $ok += $m
        Add-Content logs_q35.txt "VERIFIED $($m.tag)"
    } else {
        Add-Content logs_q35.txt "SKIP $($m.tag) weights incomplete"
    }
}

foreach ($m in $ok) {
    Add-Content logs_q35.txt "=== E1 $($m.tag)"
    & $py -m contagion.run_forced --model $m.tag --n-tasks 30 --batch-size $m.bs `
        --out "results/forced_$($m.tag).jsonl" 2>&1 |
        Select-Object -Last 1 | Add-Content logs_q35.txt
}

# Escalation and closed loop use the smallest verified driver and the largest
# verified rescuer, so the block degrades gracefully if 4B never lands.
if ($ok.Count -ge 2) {
    $small = $ok[0].tag
    $large = $ok[-1].tag
    Add-Content logs_q35.txt "=== free $small -> $large"
    & $py -m contagion.run_free --small $small --large $large `
        --policies none,random,front,conf,dispatch --budgets 0.25 --n-tasks 30 `
        --out "results/free_${small}_${large}.jsonl" 2>&1 |
        Select-Object -Last 1 | Add-Content logs_q35.txt
    Add-Content logs_q35.txt "=== closed $large"
    & $py -m contagion.run_closed --model $large --n-tasks 30 `
        --out "results/closed_$large.jsonl" 2>&1 |
        Select-Object -Last 1 | Add-Content logs_q35.txt
}

Add-Content logs_q35.txt "Q35 BLOCK DONE"
[KQ.PW]::SetThreadExecutionState([uint32]2147483648) | Out-Null
