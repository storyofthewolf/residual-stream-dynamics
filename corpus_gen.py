"""
corpus_gen.py — Generate a structured prompt corpus with contrast pairs.

Each entry has:
  - pair_id:   shared ID linking a base prompt to its contrast
  - role:      "base" or "contrast"
  - category:  prompt category (pattern, syntactic, predictability, arithmetic)
  - prompt:    the text

The contrast pair design lets entropy_analysis.py and sae_analysis.py
compare internal model states between prompts that share a stem but
diverge in expected predictability.

Usage:
    python corpus_gen.py                        # saves to corpus.json
    python corpus_gen.py --output my_corpus.json
    python corpus_gen.py --list-categories
"""

import json
import argparse
from pathlib import Path

# ============================================================================
# PROMPT PAIRS
# Each entry: (base_prompt, contrast_prompt, category, description)
# Base = higher predictability / more structured
# Contrast = lower predictability / breaks the pattern
# ============================================================================

PROMPT_PAIRS = [

    # --- Pattern continuation ---
    # Base: strong sequential pattern, model should be confident
    # Contrast: pattern breaks or becomes semantically incoherent

    ("one two three four", "one two three purple",
     "pattern", "number sequence vs color intrusion"),

    ("Monday Tuesday Wednesday Thursday", "Monday Tuesday Wednesday coffee",
     "pattern", "weekday sequence vs noun intrusion"),

    ("January February March April", "January February March democracy",
     "pattern", "month sequence vs abstract noun intrusion"),

    ("A B C D", "A B C seven",
     "pattern", "letter sequence vs digit intrusion"),

    ("red blue red blue red", "red blue red blue philosophy",
     "pattern", "color alternation vs abstract intrusion"),

    ("10 20 30 40", "10 20 30 banana",
     "pattern", "counting by tens vs noun intrusion"),

    # --- Syntactic structure ---
    # Base: grammatically well-formed, high completion certainty
    # Contrast: same stem, grammatically odd or semantically incoherent

    ("The cat sat on the", "The cat sat on democracy",
     "syntactic", "concrete location vs abstract noun"),

    ("She opened the door and", "She opened the door philosophy",
     "syntactic", "coherent continuation stem vs abrupt abstract"),

    ("The dog barked at the", "The dog barked at seventeen",
     "syntactic", "plausible object vs number"),

    ("He picked up the heavy", "He picked up the heavy equation",
     "syntactic", "adjective before concrete noun vs abstract noun"),

    ("The sun rises in the", "The sun rises in the algorithm",
     "syntactic", "directional completion vs technical noun"),

    ("They walked into the dark", "They walked into the dark calculus",
     "syntactic", "adjective before concrete vs discipline name"),

    # --- High predictability stems ---
    # Base: cultural / factual completion most readers would agree on
    # Contrast: same stem, unexpected or low-probability continuation

    ("To be or not to", "To be or not to dance",
     "predictability", "Shakespeare completion vs verb substitution"),

    ("Once upon a", "Once upon a theorem",
     "predictability", "fairy tale opener vs mathematical term"),

    ("In the beginning", "In the beginning calculus",
     "predictability", "Genesis opener vs technical term"),

    ("It was a dark and stormy", "It was a dark and stormy equation",
     "predictability", "gothic fiction stem vs math term"),

    ("The quick brown fox jumps over the lazy", "The quick brown fox jumps over the lazy theorem",
     "predictability", "pangram completion vs abstract noun"),

    # --- Arithmetic surface form ---
    # Base: standard arithmetic expression
    # Contrast: same numbers, nonsensical operator arrangement
    # (We're not testing correctness — we're studying the forward pass
    #  on structured vs unstructured numeric token sequences)

    ("1 + 1 =", "1 + 1 philosophy",
     "arithmetic", "arithmetic expression vs noun after operator"),

    ("2 + 2 =", "2 + 2 river",
     "arithmetic", "arithmetic expression vs concrete noun"),

    ("3 x 3 =", "3 x 3 democracy",
     "arithmetic", "multiplication vs abstract noun"),

    ("10 - 5 =", "10 - 5 Thursday",
     "arithmetic", "subtraction vs weekday"),

    ("5 + 3 =", "5 + 3 elephant",
     "arithmetic", "addition vs animal"),

    # --- Repetition / local coherence ---
    # Base: direct repetition (very low entropy expected)
    # Contrast: near-repetition with substitution

    ("the the the the", "the the the philosophy",
     "repetition", "pure repetition vs noun break"),

    ("go go go go", "go go go democracy",
     "repetition", "verb repetition vs abstract noun"),

    ("yes yes yes yes", "yes yes yes calculus",
     "repetition", "affirmation repetition vs technical term"),
]


def build_corpus(pairs=PROMPT_PAIRS):
    """Convert raw pairs list into structured corpus entries."""
    corpus = []
    for pair_id, (base, contrast, category, description) in enumerate(pairs):
        corpus.append({
            "pair_id": pair_id,
            "role": "base",
            "category": category,
            "description": description,
            "prompt": base,
        })
        corpus.append({
            "pair_id": pair_id,
            "role": "contrast",
            "category": category,
            "description": description,
            "prompt": contrast,
        })
    return corpus


def list_categories(corpus):
    categories = {}
    for entry in corpus:
        if entry["role"] != "base":
            continue
        cat = entry["category"]
        categories.setdefault(cat, [])
        categories[cat].append(f"  [{entry['pair_id']:02d}] {entry['description']}")
    for cat, items in categories.items():
        print(f"\n{cat} ({len(items)} pairs):")
        for item in items:
            print(item)
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate prompt corpus with contrast pairs")
    parser.add_argument("--output", type=str, default="corpus.json",
                        help="Output file path (default: corpus.json)")
    parser.add_argument("--list-categories", action="store_true",
                        help="Print corpus categories and exit")
    args = parser.parse_args()

    corpus = build_corpus()

    if args.list_categories:
        list_categories(corpus)
        return

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2)

    n_pairs = len(corpus) // 2
    categories = sorted(set(e["category"] for e in corpus))
    print(f"\nCorpus saved to {output_path}")
    print(f"  {n_pairs} contrast pairs ({len(corpus)} total prompts)")
    print(f"  Categories: {', '.join(categories)}")
    print(f"\nRun with --list-categories to see all pairs.")
    print(f"Feed to: python entropy_analysis.py --corpus {output_path}")
    print(f"         python sae_analysis.py --corpus {output_path}\n")


if __name__ == "__main__":
    main()
