"""
Flexible model & SAE loader with auto-caching.

Models are automatically downloaded and cached to ~/.cache/huggingface/hub/ on first use.
Subsequent loads are fast (loads from cache).

Usage:
    # Load model (auto-caches on first run, loads from cache afterwards)
    python predict.py "2 + 2 =" --model pythia-1b
    
    # Load model + SAE
    python predict.py "2 + 2 =" --model pythia-1b --load-sae
    
    # List available models
    python setup.py --list
    
    # Check what's cached locally
    python setup.py --list-cache
"""

import argparse
import warnings
import logging
import torch
import os
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning, module="sae_lens")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sae_lens")
warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import get_token

# ============================================================================
# MODEL REGISTRY: Define all supported models and their SAEs
# ============================================================================

MODEL_CONFIGS = {
    "pythia-160m": {
        "hf_name": "EleutherAI/pythia-160m-deduped",
        "sae_release": "sae_bench_pythia160m_sweep_standard_ctx128_0712",
        "default_layer": 2,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
        "description": "Pythia 160M (deduped)",
        "gated": False,
        "requires_auth": False
    },
    "pythia-1b": {
        "hf_name": "EleutherAI/pythia-1b-deduped",
        "sae_release": "sae_bench_pythia1b_sweep_standard_ctx128_0712",
        "default_layer": 6,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
        "description": "Pythia 1B (deduped)",
        "gated": False,
        "requires_auth": False
    },
    "pythia-2.8b": {
        "hf_name": "EleutherAI/pythia-2.8b-deduped",
        "sae_release": None,
        "default_layer": 8,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
        "description": "Pythia 2.8B (deduped)",
        "gated": False,
        "requires_auth": False
    },
    "pythia-6.9b": {
        "hf_name": "EleutherAI/pythia-6.9b-deduped",
        "sae_release": None,
        "default_layer": 12,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
        "description": "Pythia 6.9B (deduped)",
        "gated": False,
        "requires_auth": False
    },
    "llama-3.2-1b": {
        "hf_name": "meta-llama/Llama-3.2-1B",
        "sae_release": None,
        "default_layer": 6,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
        "description": "Llama 3.2 1B",
        "gated": True,
        "requires_auth": True
    },
    "llama-3.2-3b": {
        "hf_name": "meta-llama/Llama-3.2-3B",
        "sae_release": None,
        "default_layer": 8,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
        "description": "Llama 3.2 3B",
        "gated": True,
        "requires_auth": True
    },
    "gemma-2b": {
        "hf_name": "google/gemma-2b",
        "sae_release": "gemma-2b-res-jb",
        "default_layer": 6,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
        "description": "Gemma 2B",
        "gated": True,
        "requires_auth": True
    },
    "gemma-2b-it": {
        "hf_name": "google/gemma-2b-it",
        "sae_release": "gemma-2b-it-res-jb",
        "default_layer": 12,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
        "description": "Gemma 2B Instruct",
        "gated": True,
        "requires_auth": True
    },
    "gpt2-small": {
        "hf_name": "gpt2",  # TransformerLens expects "gpt2", not "gpt2-small"
        "sae_release": "gpt2-small-res-jb",
        "default_layer": 6,
        "hook_pattern": "blocks.{layer}.hook_resid_pre",
        "description": "GPT-2 Small",
        "gated": False,
        "requires_auth": False
    },
    "gpt2-medium": {
        "hf_name": "gpt2-medium",
        "sae_release": "gpt2-medium-res-jb",
        "default_layer": 8,
        "hook_pattern": "blocks.{layer}.hook_resid_pre",
        "description": "GPT-2 Medium",
        "gated": False,
        "requires_auth": False
    },
}


def load_model_and_sae(
    model_name: str,
    layer: Optional[int] = None,
    device: Optional[str] = None,
    load_sae: bool = False
) -> Tuple[HookedTransformer, Optional[SAE], dict]:
    """Load a model and optionally an SAE."""
    
    if model_name not in MODEL_CONFIGS:
        print(f"✗ Model '{model_name}' not found.")
        print(f"Available: {', '.join(MODEL_CONFIGS.keys())}")
        raise ValueError(f"Unknown model: {model_name}")
    
    cfg_dict = MODEL_CONFIGS[model_name]
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "cpu"
            print("⚠ MPS available but using CPU for stability (MPS has issues with some SAE ops)")
        else:
            device = "cpu"
    
    sae_layer = layer if layer is not None else cfg_dict["default_layer"]
    
    if load_sae and layer is None:
        print(f"ℹ SAE layer not specified; using default layer {sae_layer} for {model_name}")
    
    hook_name = cfg_dict["hook_pattern"].format(layer=sae_layer)
    
    print("\n" + "="*60)
    print(f"Loading: {cfg_dict['description']}")
    print(f"Device: {device}")
    mode = "Model + SAE" if load_sae else "Model only (no SAE)"
    print(f"Mode: {mode}")
    print(f"Cache: ~/.cache/huggingface/hub/")
    print("="*60)
    
    print("Loading model from HuggingFace...")
    try:
        model = HookedTransformer.from_pretrained(
            cfg_dict["hf_name"],
            device=device,
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub/")
        )
        print(f"✓ Model loaded")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    
    sae = None
    
    if load_sae:
        sae_release = cfg_dict.get("sae_release")
        
        if sae_release is None:
            print(f"⚠ No pre-trained SAE available for {model_name} at layer {sae_layer}")
        else:
            print(f"Loading SAE from {sae_release} (layer {sae_layer})...")
            try:
                if cfg_dict.get("requires_auth", False):
                    token = get_token()
                    if token is None:
                        print(f"⚠ {model_name} requires HuggingFace authentication.")
                        print(f"  Run: huggingface-cli login")
                        raise RuntimeError("Not authenticated")
                
                # SAELens v6 API: from_pretrained(release, sae_id) -> SAE
                # sae_id is the hook point path within the release (e.g. "blocks.6.hook_resid_post")
                sae = SAE.from_pretrained(
                    release=sae_release,
                    sae_id=hook_name,
                    device=device,
                )
                print(f"✓ SAE loaded (layer {sae_layer})")
            except Exception as e:
                print(f"✗ Error loading SAE: {e}")
                raise
    
    result_cfg = {
        "model_name": model_name,
        "layer": sae_layer,
        "hook_name": hook_name,
        "device": device,
        "sae_release": cfg_dict.get("sae_release"),
        "hf_name": cfg_dict["hf_name"],
        "load_sae": load_sae,
    }
    
    print(f"✓ Ready!\n")
    
    return model, sae, result_cfg

def list_available_models():
    print("\nAvailable Models:")
    print("-" * 70)
    for name, cfg in MODEL_CONFIGS.items():
        sae_status = "✓ SAE" if cfg.get("sae_release") else "✗ No SAE"
        auth = " (requires auth)" if cfg.get("requires_auth") else ""
        print(f"  {name:18} {sae_status:12} {cfg['description']}{auth}")
    print()
    return

def list_cached_models():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return

    subdirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    if not subdirs:
        print("Cache is empty")
        return

    print(f"\nCached models in {cache_dir}:")
    for d in sorted(subdirs):
        print(f"  {d}")
    print()
    return

"""
def list_available_saes(model_key: str):
    from sae_lens import pretrained_saes

    if model_key not in MODEL_CONFIGS:
        print(f"Unknown model '{model_key}'. Available: {', '.join(MODEL_CONFIGS.keys())}")
        return
    release = MODEL_CONFIGS[model_key].get("sae_release")
    if release is None:
        print(f"No SAE release configured for {model_key}")
        return
    all_saes = pretrained_saes.get_pretrained_saes_directory()
    if release not in all_saes:
        print(f"Release '{release}' not found in SAELens registry.")
        print("Run without --list-saes and check your SAELens version, or browse:")
        print("  https://github.com/decoderesearch/SAELens/blob/main/sae_lens/pretrained_saes.yaml")
        return
    print(f"\nAvailable sae_ids for release '{release}':")
    for sae_id in sorted(all_saes[release].saes_map.keys()):
        print(f"  {sae_id}")
    print()
    return
"""


def main():
    parser = argparse.ArgumentParser(description="Load models & SAEs")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--list-cache", action="store_true", help="Show cached models")
    #parser.add_argument("--list-saes", type=str, metavar="MODEL", help="List valid SAE releases & sae_ids for a model (e.g. --list-saes gpt2-small)")
    parser.add_argument("--model", type=str, default="pythia-1b", help="Model to load")
    parser.add_argument("--layer", type=int, default=None, help="SAE layer")
    parser.add_argument("--load-sae", action="store_true", help="Load SAE too")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
        
    if args.list_cache:
        list_cached_models()
        return
    
    #if args.list_saes:
    #    list_available_saes(args.list_saes)
    #    return
    
    try:
        model, sae, cfg = load_model_and_sae(
            args.model,
            layer=args.layer,
            load_sae=args.load_sae
        )
        print(f"Config: {cfg}")
    except Exception as e:
        print(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    main()
