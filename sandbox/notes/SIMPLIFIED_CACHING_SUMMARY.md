# Simplified Setup: Auto-Caching Only

## What Changed

Removed all manual download/storage complexity. Now the setup is **pure caching**:

- ✅ **Removed**: `--download-all` command
- ✅ **Removed**: `--model-dir` / custom storage directory option
- ✅ **Removed**: `download_model()` function
- ✅ **Removed**: `snapshot_download` import (not needed)
- ✅ **Kept**: Auto-caching to `~/.cache/huggingface/hub/` (automatic)

## How It Works Now

**You do nothing. Caching is automatic.**

```bash
# First run: downloads model (1-2 minutes depending on size)
python predict.py "2 + 2 =" --model pythia-1b

# Second run: loads from cache (5 seconds)
python predict.py "2 + 2 =" --model pythia-1b

# Every run after: cached (5 seconds)
python predict.py "2 + 2 =" --model pythia-1b
```

That's it.

## Available Commands

```bash
# Load model (model auto-caches on first run)
python predict.py "2 + 2 =" --model pythia-1b

# Load model + SAE
python predict.py "2 + 2 =" --model pythia-1b --load-sae

# Override default layer for SAE
python predict.py "2 + 2 =" --model pythia-1b --layer 8 --load-sae

# List available models
python setup.py --list

# See what's cached locally
python setup.py --list-cache

# Override device (optional)
python predict.py "2 + 2 =" --model pythia-1b --device cpu
```

## Where Models Are Cached

All models automatically go to: `~/.cache/huggingface/hub/`

Check what's cached:
```bash
python setup.py --list-cache
```

Clear cache if needed:
```bash
rm -rf ~/.cache/huggingface/hub/
```

## Your Actual Workflow

### Day 1: First time using a model
```bash
python predict.py "2 + 2 =" --model pythia-1b
# Downloads model to cache (1-2 minutes)

python step1_logit_lens.py --model pythia-1b
# Uses cached model (fast)

python step2_max_activating.py --model pythia-1b
# Uses cached model (fast)
```

### Day 2+: Everything is cached
```bash
python predict.py "2 + 2 =" --model pythia-1b
# Loads from cache (5 seconds)

python step1_logit_lens.py --model pythia-1b
# Uses cached model (5 seconds)
```

## Function Signature

```python
def load_model_and_sae(
    model_name: str,
    layer: Optional[int] = None,
    device: Optional[str] = None,
    load_sae: bool = False
) -> Tuple[HookedTransformer, Optional[SAE], dict]:
```

Parameters:
- `model_name` (required) — which model to load
- `layer` (optional) — SAE layer (defaults to model's default)
- `device` (optional) — cuda/cpu/mps (defaults to auto-select)
- `load_sae` (optional) — load SAE? (defaults to False)

Returns:
- `model` — the loaded TransformerLens model
- `sae` — the loaded SAE (or None if not loaded)
- `config` — dict with metadata (model_name, layer, hook_name, device, etc.)

## Simple Example Usage

```python
from setup_with_caching import load_model_and_sae

# Load model only (fast, no SAE)
model, sae, cfg = load_model_and_sae("pythia-1b")
print(f"Loaded {cfg['model_name']} on {cfg['device']}")

# Load model + SAE
model, sae, cfg = load_model_and_sae("pythia-1b", load_sae=True)
print(f"SAE at layer {cfg['layer']}")

# Custom layer
model, sae, cfg = load_model_and_sae("pythia-1b", layer=8, load_sae=True)
```

## Performance

| Scenario | Time |
|----------|------|
| First load (download) | 1-2 minutes |
| Subsequent loads (cached) | ~5 seconds |
| File downloaded to cache | ~/.cache/huggingface/hub/ |

## What You Don't Have to Do

- ✅ No manual downloads
- ✅ No managing storage directories
- ✅ No worrying about backups
- ✅ No environment variables
- ✅ No configuration

Just load models and they auto-cache.

## Troubleshooting

**Q: Models are downloading every time?**
A: Something cleared your cache. They'll re-download once and then cache again.

**Q: How much disk space will this use?**
A:
- pythia-160m: 0.3 GB
- pythia-1b: 2.2 GB
- pythia-3b: 6.5 GB
- gpt2-small: 0.5 GB

Total for all open models: ~11 GB

**Q: Can I delete the cache?**
A: Yes: `rm -rf ~/.cache/huggingface/hub/` — Models will re-download when needed.

---

**This is the simplest, cleanest version. Just load models and let caching do the rest.**
