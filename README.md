python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -e ".[analysis,dev]"

neuro-inspired-maxcut run configs/experiments/maxcut_baseline.yaml
