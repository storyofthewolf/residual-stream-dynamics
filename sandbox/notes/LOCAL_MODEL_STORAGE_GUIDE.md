# Local Model Storage: Download Once, Use Forever

You can download models once and store them locally on your hard drive. This is faster (no network after first download) and essential for reproducible research.

## How It Works

TransformerLens automatically caches downloaded models in `~/.cache/huggingface/hub/`. Once downloaded, subsequent loads are much faster (from disk, not network).

## Quick Start

### Option 1: Automatic Caching (Default, Recommended)

Just load models normally—they're automatically cached:

```bash
# First load: downloads and caches
python predict.py "2 + 2 =" --model pythia-1b

# Second load: loads from cache (much faster!)
python predict.py "2 + 2 =" --model pythia-1b
```

Cache location: `~/.cache/huggingface/hub/`

### Option 2: Custom Local Directory

Store models in a specific directory on your hard drive:

```bash
# Create directory
mkdir ~/models/sae_research

# Load with custom path (downloads once, then reuses)
python predict.py "2 + 2 =" --model pythia-1b --model-dir ~/models/sae_research
```

All future runs with same `--model-dir` will use that directory.

### Option 3: Pre-Download Everything

Download all open models before you start work (useful for offline work or batch jobs):

```bash
# Download all models to default cache
python setup.py --download-all

# Or to custom directory
python setup.py --download-all --model-dir ~/models/sae_research
```

## Check What's Cached

```bash
# See all cached models and their sizes
python setup.py --list-cache
```

Example output:
```
HuggingFace cache directory: /home/user/.cache/huggingface/hub

Cached models:

  EleutherAI/pythia-160m-deduped                   0.32 GB
  EleutherAI/pythia-1b-deduped                     2.20 GB
  EleutherAI/pythia-3b-deduped                     6.50 GB
  openai/gpt2-small                                0.50 GB
```

## Permanent Storage Setup

For your interpretability work, I recommend:

### Step 1: Create a dedicated directory
```bash
mkdir -p ~/models/sae_research
cd ~/models/sae_research
```

### Step 2: Download all open models (no auth needed)
```bash
python setup.py --download-all --model-dir ~/models/sae_research
```

This downloads:
- Pythia 160M (0.3 GB)
- Pythia 1B (2.2 GB)
- Pythia 3B (6.5 GB)
- GPT-2 Small (0.5 GB)
- GPT-2 Medium (1.5 GB)

**Total: ~11 GB**

### Step 3: Use in your scripts

Update your `predict.py` and pipeline scripts:

```python
from pathlib import Path
from setup import load_model_and_sae

MODEL_DIR = Path.home() / "models" / "sae_research"

# In your loading code:
model, sae, cfg = load_model_and_sae(
    "pythia-1b",
    model_dir=MODEL_DIR
)
```

Or via command line:

```bash
python predict.py "2 + 2 =" --model pythia-1b --model-dir ~/models/sae_research
python step1_logit_lens.py --model pythia-1b --model-dir ~/models/sae_research
```

## Understanding the Cache Structure

HuggingFace stores models in a special folder structure. Don't manually move files—let the tools handle it.

```
~/.cache/huggingface/hub/
├── models--EleutherAI--pythia-160m-deduped/
│   ├── snapshots/
│   │   └── [commit-hash]/
│   │       ├── config.json
│   │       ├── pytorch_model.bin
│   │       └── ...
│   └── blobs/  (actual model files)
│
├── models--EleutherAI--pythia-1b-deduped/
│   └── ...
└── ...
```

If you need to clean up:

```bash
# Delete specific model
rm -rf ~/.cache/huggingface/hub/models--EleutherAI--pythia-1b-deduped/

# See total size
du -sh ~/.cache/huggingface/hub/

# Clear entire cache (re-download later if needed)
rm -rf ~/.cache/huggingface/hub/
```

## Performance Comparison

**First load (download from network):**
- Pythia 1B: ~1-2 minutes (depends on internet speed)

**Subsequent loads (from cache):**
- Pythia 1B: ~5-10 seconds

**With custom SSD directory:**
- Even faster: ~2-5 seconds

## Custom Directory for Offline Work

If you have limited network access or want to guarantee offline operation:

```bash
# On a computer with internet:
python setup.py --download-all --model-dir ~/models/sae_research

# Copy the models directory to your target machine
# (e.g., via external drive, USB, or transfer)

# On target machine, use:
python predict.py "2 + 2 =" --model pythia-1b --model-dir ~/models/sae_research
# Works completely offline!
```

## For Your SAELens Pipeline

Update your three scripts to use local models:

**setup.py configuration (add to top):**
```python
from pathlib import Path

# Option 1: Use default cache (automatic)
MODEL_DIR = None  # Uses ~/.cache/huggingface/hub

# Option 2: Use custom directory (permanent local storage)
MODEL_DIR = Path.home() / "models" / "sae_research"

def load_model(model_name: str, layer: Optional[int] = None):
    from setup import load_model_and_sae
    return load_model_and_sae(
        model_name,
        layer=layer,
        load_sae=True,
        model_dir=MODEL_DIR
    )
```

**In your scripts:**
```bash
# Use same directory for all experiments
python step1_logit_lens.py --model pythia-1b --model-dir ~/models/sae_research
python step2_max_activating.py --model pythia-1b --model-dir ~/models/sae_research
python step3_causal_patch.py --model pythia-1b --model-dir ~/models/sae_research

# Easy to switch models
python step1_logit_lens.py --model pythia-3b --model-dir ~/models/sae_research
python step1_logit_lens.py --model gpt2-small --model-dir ~/models/sae_research
```

## Approximate Disk Requirements

| Model | Size | Notes |
|-------|------|-------|
| pythia-160m | 0.3 GB | Minimal |
| pythia-1b | 2.2 GB | Good balance |
| pythia-3b | 6.5 GB | Larger |
| gpt2-small | 0.5 GB | Tiny |
| gpt2-medium | 1.5 GB | Small |
| gemma-2b | 5.2 GB | (if authenticated) |
| llama-3.2-1b | 2.5 GB | (if authenticated) |
| llama-3.2-3b | 7.5 GB | (if authenticated) |

**Recommendation:** Start with Pythia 1B + 3B + GPT-2 small (~9 GB total). Add others as needed.

## Benefits of Local Storage

✓ **Speed** — No network I/O after first download  
✓ **Reproducibility** — Same exact models for all experiments  
✓ **Offline work** — Run without internet  
✓ **Portability** — Copy directory between machines  
✓ **Backup** — Easy to backup or restore entire model set  
✓ **Version control** — Keep specific versions long-term  

## Troubleshooting

### Models are in default cache, want to move to custom directory

Don't manually move. Re-download to new location:
```bash
python setup.py --download-all --model-dir ~/models/sae_research
```

The cache deduplicates, so re-downloading is quick (skips already-downloaded files).

### "No space left on device"

Check disk usage:
```bash
du -sh ~/.cache/huggingface/hub/
du -sh ~/models/sae_research/
```

Remove unused models:
```bash
rm -rf ~/.cache/huggingface/hub/models--google--gemma*
```

### Want to verify downloaded models are complete

```bash
# Check model loads successfully
python predict.py "Hello" --model pythia-1b --model-dir ~/models/sae_research

# If it loads without error, the download is complete
```

---

## My Recommendation

For your interpretability work:

```bash
# 1. Create local storage
mkdir -p ~/models/sae_research

# 2. Download models you'll use repeatedly
python setup.py --download-all --model-dir ~/models/sae_research

# 3. Use in your pipeline
python step1_logit_lens.py --model pythia-1b --model-dir ~/models/sae_research

# 4. You're done—fast, offline, reproducible
```

Then you have all models stored locally, can work offline, and get fast loading times forever.
