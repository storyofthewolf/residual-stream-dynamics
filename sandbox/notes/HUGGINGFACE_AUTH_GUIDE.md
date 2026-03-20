# HuggingFace Authentication for Gated Models

Llama and Gemma models are **gated** on HuggingFace, meaning they require authentication to download and use.

## Quick Start: 5 Minutes

### Step 1: Create a HuggingFace Account (if needed)
Go to https://huggingface.co/join and sign up (free).

### Step 2: Accept Model Licenses

Visit each model page and click **"Agree and access repository"**:
- https://huggingface.co/meta-llama/Llama-3.2-3B
- https://huggingface.co/meta-llama/Llama-3.2-1B
- https://huggingface.co/google/gemma-2b
- https://huggingface.co/google/gemma-2b-it

This tells Meta/Google you agree to their licensing terms.

### Step 3: Create an API Token
1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Give it a name (e.g., "SAELens")
4. Set role to **"Read"** (not write)
5. Click **"Create token"**
6. Copy the token (starts with `hf_...`)

### Step 4: Authenticate Locally

**Option A: Interactive (Recommended)**
```bash
huggingface-cli login
```
Then paste your token when prompted. It will be saved to `~/.huggingface/token`.

**Option B: Environment Variable (One-time)**
```bash
export HF_TOKEN="hf_your_token_here"
python predict.py --model llama-3.2-3b
```

**Option C: Programmatic (Python)**
```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

### Step 5: Verify It Works
```bash
python predict.py --list-models
```

You should see:
```
  llama-3.2-3b         ✓ SAE    🔓 Gated (authenticated)  Llama 3.2 3B ...
```

Instead of:
```
  llama-3.2-3b         ✓ SAE    🔒 Gated (login required)  Llama 3.2 3B ...
```

---

## Troubleshooting

### Error: "Access to model meta-llama/Llama-3.2-3B is restricted"

**Solution:** You forgot step 2. Visit https://huggingface.co/meta-llama/Llama-3.2-3B and click **"Agree and access repository"**.

### Error: "Token is invalid"

**Solution:** Check your token:
1. Go to https://huggingface.co/settings/tokens
2. Make sure your token still exists (not revoked)
3. Copy again and re-authenticate: `huggingface-cli login`

### Error: "huggingface-cli not found"

**Solution:** Install HuggingFace CLI:
```bash
pip install huggingface-hub --break-system-packages
```

Then try again:
```bash
huggingface-cli login
```

### Still getting "Access denied"

**Solution:** Sometimes there's a delay. Try:
```bash
# Clear cached credentials
rm -rf ~/.huggingface/

# Re-authenticate
huggingface-cli login

# Wait 1-2 minutes, then try again
python predict.py --list-models
```

---

## Models Requiring Authentication

| Model | Link | Requirement |
|-------|------|-------------|
| Llama 3.2 1B | https://huggingface.co/meta-llama/Llama-3.2-1B | Accept Meta license |
| Llama 3.2 3B | https://huggingface.co/meta-llama/Llama-3.2-3B | Accept Meta license |
| Gemma 2B | https://huggingface.co/google/gemma-2b | Accept Google license |
| Gemma 2B-IT | https://huggingface.co/google/gemma-2b-it | Accept Google license |

## Models NOT Requiring Authentication

These are open and work immediately:
- Pythia (all sizes)
- GPT-2 (small, medium)

---

## Alternative: Use Open Models Only

If you don't want to authenticate, stick with these:

```bash
python predict.py "2 + 2 =" --model pythia-1b          # Open
python predict.py "2 + 2 =" --model pythia-3b          # Open
python predict.py "2 + 2 =" --model gpt2-small         # Open
python predict.py "2 + 2 =" --model gpt2-medium        # Open
```

These have pre-trained SAEs and don't require any authentication.

---

## Why Models Are Gated

Meta and Google gate Llama and Gemma because:
1. **Research purposes** — they want to track usage
2. **Responsible deployment** — accepting terms shows you understand risks
3. **Compliance** — meeting licensing and export requirements

It's quick to accept (1-2 minutes) and then you have permanent access.
