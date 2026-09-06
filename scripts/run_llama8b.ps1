# Llama 3.1 8B Instruct: gated download, load verification, then the E1
# measurement at 8 bit so it fits the card. The token is read from the
# environment and is never written to disk by this script.
$py = "$env:USERPROFILE\miniconda3\envs\gemma_spectral\python.exe"
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
Set-Location "$PSScriptRoot/.."

Add-Type -Name PL -Namespace KL -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[KL.PL]::SetThreadExecutionState([uint32]2147483649) | Out-Null

$repo = "meta-llama/Llama-3.1-8B-Instruct"
$verified = $false
foreach ($try in 1..5) {
    Add-Content logs_llama8b.txt "download attempt $try"
    & $py -c @"
from huggingface_hub import snapshot_download
snapshot_download('$repo')
"@ 2>&1 | Select-Object -Last 1 | Add-Content logs_llama8b.txt

    & $py -c @"
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('$repo', dtype=torch.bfloat16, device_map='cpu')
print('VERIFIED', round(sum(p.numel() for p in m.parameters())/1e9, 2), 'B')
"@ 2>&1 | Select-String "VERIFIED" | Add-Content logs_llama8b.txt
    if ($LASTEXITCODE -eq 0) { $verified = $true; break }
    Start-Sleep 30
}

if (-not $verified) {
    Add-Content logs_llama8b.txt "LLAMA8B UNAVAILABLE after retries"
    [KL.PL]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    exit 0
}

$env:HF_HUB_OFFLINE = "1"
Add-Content logs_llama8b.txt "=== E1 llama-8b at 8 bit"
& $py -m contagion.run_forced --model llama-8b@8bit --n-tasks 30 --batch-size 8 `
    --out results/forced_llama-8b-8bit.jsonl 2>&1 |
    Select-Object -Last 2 | Add-Content logs_llama8b.txt

Add-Content logs_llama8b.txt "LLAMA8B DONE"
[KL.PL]::SetThreadExecutionState([uint32]2147483648) | Out-Null
