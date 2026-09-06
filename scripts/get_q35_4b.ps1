# Retry the Qwen3.5-4B download (the first attempt lost a shard when the
# laptop slept), verify the weights load, then run the 4B experiments once
# the running Qwen3.5 block has released the GPU.
$py = "$env:USERPROFILE\miniconda3\envs\gemma_spectral\python.exe"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
Remove-Item Env:\HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Set-Location "$PSScriptRoot/.."

Add-Type -Name PD -Namespace KD -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[KD.PD]::SetThreadExecutionState([uint32]2147483649) | Out-Null

$verified = $false
foreach ($try in 1..5) {
    Add-Content logs_q35_4b.txt "download attempt $try"
    & $py -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')" 2>&1 |
        Select-Object -Last 1 | Add-Content logs_q35_4b.txt
    & $py -c "from transformers import AutoModelForCausalLM; import torch; m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', dtype=torch.bfloat16, device_map='cpu'); print('VERIFIED', round(sum(p.numel() for p in m.parameters())/1e9,2))" 2>&1 |
        Select-String "VERIFIED" | Add-Content logs_q35_4b.txt
    if ($LASTEXITCODE -eq 0) { $verified = $true; break }
    Start-Sleep 30
}

if (-not $verified) {
    Add-Content logs_q35_4b.txt "4B UNAVAILABLE after retries"
    [KD.PD]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    exit 0
}

# Wait for the running Qwen3.5 block to finish before taking the GPU.
while (-not (Select-String -Path logs_q35.txt -Pattern "Q35 BLOCK DONE" -Quiet)) {
    Start-Sleep 120
}
Start-Sleep 30

$env:HF_HUB_OFFLINE = "1"
Add-Content logs_q35_4b.txt "=== E1 qwen3.5-4b"
& $py -m contagion.run_forced --model qwen3.5-4b --n-tasks 30 --batch-size 12 `
    --out results/forced_qwen3.5-4b.jsonl 2>&1 | Select-Object -Last 1 | Add-Content logs_q35_4b.txt

Add-Content logs_q35_4b.txt "=== free qwen3.5-0.8b -> qwen3.5-4b"
& $py -m contagion.run_free --small qwen3.5-0.8b --large qwen3.5-4b `
    --policies none,random,front,conf,dispatch --budgets 0.25 --n-tasks 30 `
    --out results/free_qwen3.5-0.8b_qwen3.5-4b.jsonl 2>&1 |
    Select-Object -Last 1 | Add-Content logs_q35_4b.txt

Add-Content logs_q35_4b.txt "=== closed qwen3.5-4b"
& $py -m contagion.run_closed --model qwen3.5-4b --n-tasks 30 `
    --out results/closed_qwen3.5-4b.jsonl 2>&1 |
    Select-Object -Last 1 | Add-Content logs_q35_4b.txt

Add-Content logs_q35_4b.txt "Q35 4B DONE"
[KD.PD]::SetThreadExecutionState([uint32]2147483648) | Out-Null
