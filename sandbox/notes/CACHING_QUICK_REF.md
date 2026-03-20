# Model Caching: Quick Reference

## The Basics

Models are automatically cached after first download. Subsequent loads are much faster.

```bash
# First load: downloads (~1-2 min depending on model size)
python predict.py "2 + 2 =" --model pythia-1b

# Second load: from cache (~5 seconds)
python predict.py "2 + 2 =" --model pythia-1b
```

## Common Commands

### Check what's cached
```bash
python setup.py --list-cache
```

### Pre-download all models
```bash
python setup.py --download-all
```

### Use custom storage directory
```bash
python predict.py "2 + 2 =" --model pythia-1b --model-dir ~/my_models
```

### Pre-download to custom directory
```bash
python setup.py --download-all --model-dir ~/my_models
```

## Where Models Are Stored

**Default:** `~/.cache/huggingface/hub/`

**Custom:** Wherever you specify with `--model-dir`

## Typical Workflow

### One-time setup
```bash
# Create local storage
mkdir ~/models/sae_research

# Download models (10-15 minutes for all open models)
python setup.py --download-all --model-dir ~/models/sae_research
```

### In your scripts
```bash
# Always use same directory
python step1_logit_lens.py --model pythia-1b --model-dir ~/models/sae_research
python step2_max_activating.py --model pythia-1b --model-dir ~/models/sae_research
python step3_causal_patch.py --model pythia-1b --model-dir ~/models/sae_research
```

## Model Sizes (Approximate)

| Model | Size |
|-------|------|
| pythia-160m | 0.3 GB |
| pythia-1b | 2.2 GB |
| pythia-3b | 6.5 GB |
| gpt2-small | 0.5 GB |
| gpt2-medium | 1.5 GB |

**Total for all open models: ~11 GB**

## Performance

| Scenario | Load Time |
|----------|-----------|
| First load (download) | 1-2 minutes |
| From cache (default) | 5-10 seconds |
| From SSD cache | 2-5 seconds |

## With Your Pipeline

In **setup.py** or at top of scripts:

```python
from pathlib import Path
from setup import load_model_and_sae

# Option 1: Auto cache (default)
model, sae, cfg = load_model_and_sae("pythia-1b")

# Option 2: Custom directory
model, sae, cfg = load_model_and_sae(
    "pythia-1b", 
    model_dir=Path.home() / "models" / "sae_research"
)
```

Or CLI:
```bash
python predict.py "prompt" --model pythia-1b --model-dir ~/models/sae_research
```

## Troubleshooting

**Q: Models are downloading every time?**  
A: Check if they're in `~/.cache/huggingface/hub/`. If not, they were cleaned up. Re-run `python setup.py --download-all` or just use them normally (they'll re-download).

**Q: How do I move models from default cache to custom directory?**  
A: Don't move manually. Just run: `python setup.py --download-all --model-dir ~/new_location` (it'll skip what's already there)

**Q: Can I work offline?**  
A: Yes! Once downloaded, models don't need internet. Use `--model-dir ~/models/sae_research` and you're completely offline.

**Q: How much disk space do I need?**  
A: ~11 GB for all open models. Just pythia-1b is 2.2 GB.

See **LOCAL_MODEL_STORAGE_GUIDE.md** for detailed instructions.
