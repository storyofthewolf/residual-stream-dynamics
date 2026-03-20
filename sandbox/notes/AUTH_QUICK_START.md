# Quick Start: HuggingFace Authentication (2 Minutes)

If you want to use **Llama** or **Gemma** models, you need to authenticate once.

## Do This Once

```bash
# 1. Install the CLI tool (if needed)
pip install huggingface-hub --break-system-packages

# 2. Authenticate
huggingface-cli login

# 3. When prompted, paste your token (see below)
```

## Get Your Token (1 minute)

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it "SAELens"
4. Set role to **"Read"**
5. Click **"Create token"**
6. Copy and paste into the prompt above

## Accept Model Licenses

Visit these pages and click **"Agree and access repository"**:
- https://huggingface.co/meta-llama/Llama-3.2-3B
- https://huggingface.co/google/gemma-2b

(Do this even if you're already logged in—it's separate.)

## Verify It Works

```bash
python predict.py --list-models
```

You should see:
```
  llama-3.2-3b         ✓ SAE    🔓 Gated (authenticated)  Llama 3.2 3B ...
  gemma-2b             ✓ SAE    🔓 Gated (authenticated)  Gemma 2B ...
```

---

## Alternative: Use Open Models (No Auth Needed)

```bash
python predict.py "2 + 2 =" --model pythia-1b      # Works immediately
python predict.py "2 + 2 =" --model pythia-3b      # Works immediately
python predict.py "2 + 2 =" --model gpt2-small     # Works immediately
```

See `HUGGINGFACE_AUTH_GUIDE.md` for detailed troubleshooting.
