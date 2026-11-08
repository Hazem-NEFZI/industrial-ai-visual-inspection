# Installation

Python 3.10 or newer. The tested environment and exact dependency snapshot are in `VALIDATION.md` and `requirements-tested.txt`. Use a separate environment per project.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run all commands from the repository root. Core workflows require no API keys. For interactive projects install `python -m pip install -e ".[app]"`. Industrial CNN training additionally requires `python -m pip install -e ".[deep]"`. These extras exist only where relevant.

A 4 GB GPU is not needed for the core demos. Use CPU first; SAM and CNN training depend on your installed PyTorch build. No SAM weights are included. ROS 2 is a separate optional environment, not a pip dependency of the core project.
