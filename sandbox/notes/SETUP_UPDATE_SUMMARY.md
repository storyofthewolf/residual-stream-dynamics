# Updated Multi-Model Setup: Authentication Handling

## What Changed

Your `setup.py` now:

1. **Detects gated models** — Llama and Gemma are marked as requiring authentication
2. **Checks for authentication** — Before loading a gated model, it verifies you're logged in
3. **Provides clear guidance** — If auth is missing, it shows helpful instructions with links
4. **Shows model status** — `python predict.py --list-models` displays which models need auth

## Your Options

### Option A: Use Only Open Models (No Setup Needed)

These work immediately without any authentication:

```bash
python predict.py "2 + 2 =" --model pythia-1b
python predict.py "2 + 2 =" --model pythia-3b
python predict.py "2 + 2 =" --model gpt2-small
python predict.py "2 + 2 =" --model gpt2-medium
```

All have **pre-trained SAEs available**, so you can use them immediately for your interpretability work.

**Recommendation:** Start here. Pythia is designed specifically for interpretability research and these models work great for learning SAELens.

---

### Option B: Authenticate Once (5 minutes) for Llama/Gemma Access

If you want to use Llama 3.2 or Gemma models:

```bash
# 1. Get token: https://huggingface.co/settings/tokens (create new token)

# 2. Authenticate once
huggingface-cli login
# (paste token when prompted)

# 3. Accept model licenses
# Visit:
# - https://huggingface.co/meta-llama/Llama-3.2-3B (click "Agree")
# - https://huggingface.co/google/gemma-2b (click "Agree")

# 4. Verify
python predict.py --list-models
```

Then you can use:
```bash
python predict.py "2 + 2 =" --model llama-3.2-3b
python predict.py "2 + 2 =" --model gemma-2b
```

See `AUTH_QUICK_START.md` for the minimal version, or `HUGGINGFACE_AUTH_GUIDE.md` for detailed troubleshooting.

---

## Current Setup Summary

| Model | Status | SAE | Notes |
|-------|--------|-----|-------|
| pythia-160m | ✓ Open | ✓ Yes | Interpretability-focused |
| pythia-1b | ✓ Open | ✓ Yes | **Recommended for learning** |
| pythia-3b | ✓ Open | ✓ Yes | Better performance |
| gpt2-small | ✓ Open | ✓ Yes | Your original baseline |
| gpt2-medium | ✓ Open | ✓ Yes | Medium performance |
| llama-3.2-1b | 🔒 Gated | ✗ No | Better capability, requires auth |
| llama-3.2-3b | 🔒 Gated | ✗ No | Better capability, requires auth |
| gemma-2b | 🔒 Gated | ✓ Yes | Best SAE coverage, requires auth |
| gemma-2b-it | 🔒 Gated | ✓ Yes | Instruction-tuned, requires auth |

---

## My Recommendation

**For your SAELens exploration (#2: learning methodology):**

1. **Start with Pythia 1B or 3B** (no auth needed)
   - Built for interpretability research
   - Pre-trained SAEs available
   - Good enough to learn the methodology

2. **Use GPT-2 as your baseline**
   - You already have it working
   - Comparison shows capability differences

3. **Skip Llama/Gemma for now** unless you specifically need better model capability
   - Authentication adds friction
   - No benefit if you're just learning SAELens mechanics

**Example workflow:**
```bash
# Learn SAELens on Pythia (clean, no auth needed)
python step1_logit_lens.py --model pythia-1b
python step2_max_activating.py --model pythia-1b
python step3_causal_patch.py --model pythia-1b

# Once comfortable, optionally test on other models
python step1_logit_lens.py --model pythia-3b
python step1_logit_lens.py --model gpt2-small
```

---

## What to Do Now

1. **Try the open models:**
   ```bash
   python predict.py --list-models
   python predict.py "2 + 2 =" --model pythia-1b
   python predict.py "2 + 2 =" --model pythia-3b
   ```

2. **If these work:** You're ready for SAELens exploration. Move to step1, step2, step3 with `--model` flag:
   ```bash
   python step1_logit_lens.py --model pythia-1b
   ```

3. **If you want Llama/Gemma later:** Follow `AUTH_QUICK_START.md` (5-minute one-time setup)

---

## Files Included

- **setup.py** — Updated with auth handling
- **predict.py** — Token prediction (unchanged)
- **AUTH_QUICK_START.md** — Minimal guide (2 minutes)
- **HUGGINGFACE_AUTH_GUIDE.md** — Full guide with troubleshooting
- **MODEL_SELECTION_GUIDE.md** — Full documentation (original)
- **QUICK_REFERENCE.md** — Command reference (original)

---

## Next Steps for Your Pipeline

Update your `step1_logit_lens.py`, `step2_max_activating.py`, and `step3_causal_patch.py`:

At the top of each script, add:
```python
import argparse
from setup import load_model_and_sae

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pythia-1b')
parser.add_argument('--layer', type=int, default=None)
args = parser.parse_args()

model, sae, cfg = load_model_and_sae(
    args.model,
    layer=args.layer,
    load_sae=True
)

hook_name = cfg['hook_name']
# ... rest of script uses model, sae, hook_name
```

Then run:
```bash
python step1_logit_lens.py --model pythia-1b
python step1_logit_lens.py --model pythia-3b
python step1_logit_lens.py --model gpt2-small
```

This lets you quickly compare interpretability results across models without changing code.
