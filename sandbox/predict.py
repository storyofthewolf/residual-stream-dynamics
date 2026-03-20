"""
Token prediction with verbose introspection.
Supports multiple models via --model flag.

Usage:
    python predict.py "2 + 2 =" --model pythia-1b
    python predict.py "The capital of France is" --model llama-3.2-3b --nt 5
    python predict.py "Hello" --model gemma-2b --verbose
"""
# captures import of python files from one directory up
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
import torch
from setup import load_model_and_sae, list_available_models, list_cached_models

parser = argparse.ArgumentParser()
parser.add_argument('prompt', type=str, nargs='?', default=' ', help='input prompt')
parser.add_argument('--model', type=str, default='pythia-1b', help='model to use (see --list-models)')
parser.add_argument('--list', action='store_true', help='list available models and exit')
parser.add_argument("--list-cache", dest='list_cache', action="store_true", help="Show cached models")
parser.add_argument('--np', type=int, default=10, help='number of token predictions to show')
parser.add_argument('--nt', type=int, default=5, help='number of tokens to generate')
parser.add_argument('--temp', type=float, default=0.0, help='temperature for sampling')
parser.add_argument('--load-sae', dest="load_sae", action='store_true', help='load model with SAE')
parser.add_argument('--layer', type=int, default=None, help='override default layer for SAE')
parser.add_argument('--verbose', action='store_true', help='show token-by-token generation')

args = parser.parse_args()

if args.list:
    list_available_models()
    exit(0)

if args.list_cache:
    list_cached_models()
    exit(0)

# Load model
model, sae, cfg = load_model_and_sae(
    args.model,
    layer=args.layer,
    load_sae=args.load_sae
)

device = cfg['device']
npredict = args.np
ntokens = args.nt
temperature = args.temp
prompt = args.prompt

print("\n" + "="*70)
print(f"PROMPT: '{prompt}'")
print(f"MODEL: {args.model} | LAYER: {cfg['layer']} | TEMP: {temperature}")
print("="*70 + "\n")

# Tokenize prompt
tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens)

# Optional: Show logits at all positions in prompt
if False:  # Set to True to enable
    print("SECTION 1: Predictions at each position in prompt")
    print("-" * 70)
    for pos in range(tokens.shape[1]):
        pos_logits = logits[0, pos, :]
        probs = torch.softmax(pos_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5)
        print(f"Position {pos}:")
        for prob, idx in zip(top_probs, top_indices):
            token_str = model.tokenizer.decode(idx.item())
            print(f"  '{token_str}': {prob:.4f}")
        print()


# SECTION 2: Examine last position (what comes next)
print("SECTION 2: Top predictions for next token")
print("-" * 70)
next_token_logits = logits[0, -1, :]
probs = torch.softmax(next_token_logits, dim=-1)
top_probs, top_indices = torch.topk(probs, k=npredict)

print(f"Top {npredict} predictions after '{prompt}':")
for prob, idx in zip(top_probs, top_indices):
    token_str = model.tokenizer.decode(idx)
    print(f"  '{token_str}': {prob:.4f}")


# SECTION 3: Generate new tokens
print("\nSECTION 3: Token generation")
print("-" * 70)

if not args.verbose:
    # Fast path: use built-in generate
    print(f"Generating {ntokens} tokens...")
    output = model.generate(tokens, max_new_tokens=ntokens, temperature=temperature)
    generated_text = model.to_string(output)
else:
    # Verbose path: explicit loop with introspection
    print(f"Generating {ntokens} tokens (verbose)...")
    print(f"Initial tokens shape: {tokens.shape}")
    generated_text = model.to_string(tokens)
    
    for i in range(ntokens):
        print(f"\n--- Token {i+1}/{ntokens} ---")
        logits = model(tokens)
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        
        # Show top predictions
        top_probs, top_indices = torch.topk(probs, k=npredict)
        print(f"Top {npredict} predictions:")
        for prob, idx in zip(top_probs, top_indices):
            token_str = model.tokenizer.decode(idx)
            print(f"  '{token_str}': {prob:.4f}")
        
        # Sample next token
        if temperature == 0:
            next_token = torch.argmax(probs).unsqueeze(0)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        
        next_token = next_token.unsqueeze(0)  # Shape: [1, 1]
        
        generated_next = model.to_string(next_token)
        print(f"Selected: '{generated_next}'")
        
        # Append and continue
        tokens = torch.cat([tokens, next_token], dim=1)
        generated_text = model.to_string(tokens)
        print(f"Running text: '{generated_text}'")


# Final output
print("\n" + "="*70)
print("FINAL OUTPUT")
print("="*70)
print(f"Prompt:        {prompt}")
print(f"Generated:     {generated_text}")
print(f"Length:        {tokens.shape[1]} tokens total")
print("="*70 + "\n")
