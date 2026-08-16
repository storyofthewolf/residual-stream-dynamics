"""colab/build_notebook.py — Generate residual_stream_dynamics_colab.ipynb.

The notebook is generated from this script rather than hand-edited so the
cell sources stay reviewable in plain text and diff cleanly in git.

Regenerate after editing:
    python colab/build_notebook.py
"""

import json
from pathlib import Path

REPO_URL = "https://github.com/storyofthewolf/residual-stream-dynamics.git"
_HERE = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").split("\n"),
    }


CELLS = [
    md("""
# residual-stream-dynamics — Colab runner

Runs the corpus workflows on a free-tier GPU. Execute the setup cells
(1–5) in order once per session, then run whichever analysis cells you need.

**Free-tier notes**
- The GPU is a **T4 (16GB)**. `gpt2-small` through `pythia-2.8b` all fit;
  `pythia-6.9b` does **not** and is not usable here.
- The runtime is recycled after ~90 min idle / 12 h max, and `/content` is
  wiped with it. Cell 4 mounts Drive so `data/` and `figures/` survive.
- Set the runtime to GPU first: **Runtime → Change runtime type → T4 GPU**.
"""),

    md("## 1. Verify the GPU is attached"),
    code("""
!nvidia-smi

import torch
print()
print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  ({p.total_memory / 1024**3:.1f} GB)")
else:
    print("NO GPU — set Runtime > Change runtime type > T4 GPU, then rerun.")
"""),

    md("""
## 2. Clone the repository

`data/` and `figures/` are gitignored, so this clone is small (a few MB).
Results are written to Drive in cell 4.
"""),
    code(f"""
import os
from pathlib import Path

REPO_URL  = "{REPO_URL}"
REPO_NAME = "residual-stream-dynamics"
REPO_DIR  = Path("/content") / REPO_NAME

if REPO_DIR.exists():
    print(f"{{REPO_DIR}} already exists — pulling latest")
    !cd {{REPO_DIR}} && git pull --ff-only
else:
    !git clone {{REPO_URL}} {{REPO_DIR}}

os.chdir(REPO_DIR)
print(f"\\ncwd: {{Path.cwd()}}")
"""),

    md("""
## 3. Install dependencies

Colab ships its own torch build — we deliberately do **not** reinstall it, since
pip would pull a CPU-only or CUDA-mismatched wheel and break GPU support.
Only the packages Colab lacks are installed.
"""),
    code("""
# Install everything EXCEPT torch (Colab's preinstalled build is CUDA-matched).
!pip install -q transformer_lens sae_lens

import torch
print(f"\\ntorch {torch.__version__} — cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "GPU lost after install — Runtime > Restart, then rerun."
"""),

    md("""
## 4. Mount Drive and redirect outputs

`data/` and `figures/` become symlinks into Drive, so every `.npz` and `.png`
the workflows write persists across runtime restarts. Nothing in the repo code
changes — the workflows still resolve paths from `_PROJECT_ROOT`, which now
points through the symlink.
"""),
    code("""
from google.colab import drive
from pathlib import Path
import shutil, os

drive.mount("/content/drive")

# Everything lives under one Drive folder; change if you prefer another location.
DRIVE_ROOT = Path("/content/drive/MyDrive/residual-stream-dynamics")
(DRIVE_ROOT / "data").mkdir(parents=True, exist_ok=True)
(DRIVE_ROOT / "figures").mkdir(parents=True, exist_ok=True)

REPO_DIR = Path("/content/residual-stream-dynamics")

for name in ("data", "figures"):
    local = REPO_DIR / name
    target = DRIVE_ROOT / name
    if local.is_symlink():
        local.unlink()
    elif local.exists():
        # A fresh clone has an empty dir (just .gitkeep) — safe to replace.
        shutil.rmtree(local)
    local.symlink_to(target)
    print(f"{local}  ->  {target}")

print("\\nOutputs will persist in Drive across runtime restarts.")
"""),

    md("""
## 5. Smoke test

Confirms the model loads on CUDA and the pipeline runs end to end before you
commit to a long job.
"""),
    code("""
!python workflows/single_prompt.py --model gpt2-small --hooks resid_post \\
    --logit-lens --no-plots --alpha 1.0
"""),

    md("""
---
## Analysis workflows

Each cell is independent. `--device` now defaults to auto-detect, so CUDA is
picked up automatically — no flag needed.

Add `--run-tag <name>` to keep successive runs from overwriting each other.
"""),

    md("### Entropy analysis"),
    code("""
!python workflows/entropy_analysis.py --model gpt2-small --save-data
"""),

    md("### Ablation analysis (the primary experiment)"),
    code("""
!python workflows/ablation_analysis.py --model gpt2-small \\
    --ev-thresholds 0.50 0.75 0.90 0.95 0.99 \\
    --save-data
"""),

    md("### c_k spectrum analysis"),
    code("""
!python workflows/ck_analysis.py --model gpt2-small --save-data
"""),

    md("### W_U subspace analysis"),
    code("""
!python workflows/wu_subspace_analysis.py --model gpt2-small --save-data
"""),

    md("### Mechanics analysis"),
    code("""
!python workflows/mechanics_analysis.py --model gpt2-small --save-data
"""),

    md("""
---
## Sweeping models

`pythia-2.8b` and `gpt2-xl` load in float16 automatically on CUDA so they fit
the T4. `pythia-6.9b` is deliberately excluded — it does not fit in 16GB.

Run this unattended; results accumulate in Drive. If the runtime dies partway,
completed models are already saved and you can restart from the survivors.
"""),
    code("""
MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large", "pythia-160m", "pythia-1b"]

for m in MODELS:
    print(f"\\n{'='*70}\\n  {m}\\n{'='*70}")
    !python workflows/ablation_analysis.py --model {m} \\
        --ev-thresholds 0.50 0.75 0.90 0.95 0.99 --save-data --no-plots
"""),

    md("""
### Larger models (float16, one at a time)

Run these individually and restart the runtime between them to release VRAM.
"""),
    code("""
!python workflows/ablation_analysis.py --model pythia-2.8b \\
    --ev-thresholds 0.50 0.90 0.99 --save-data --no-plots
"""),

    md("""
---
## Free VRAM between models

Colab does not release GPU memory until the process exits. The `!python`
invocations above each run in their own process, so VRAM is freed automatically
when they finish. Use this only if you loaded a model inside the notebook itself.
"""),
    code("""
import gc, torch
gc.collect()
torch.cuda.empty_cache()
print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"VRAM reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
"""),

    md("""
---
## Check what has been saved to Drive
"""),
    code("""
!du -sh /content/drive/MyDrive/residual-stream-dynamics/data/* 2>/dev/null
!echo "---"
!ls -lh /content/drive/MyDrive/residual-stream-dynamics/data/ablation/ 2>/dev/null | tail -20
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = _HERE / "residual_stream_dynamics_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
