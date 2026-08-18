"""
corpus_gen_neutral.py — Non-moral control corpus for the moral-polarity experiment.

This is the CONTROL for corpus_gen_moral.py, and exists to answer one question:
are the entropy effects that corpus measures attributable to moral content, or to
the syntactic frames the moral words are embedded in?

--------------------------------------------------------------------------
Why this corpus exists
--------------------------------------------------------------------------
Running the moral corpus on gpt2-small and gpt2-medium produced large,
highly significant logit-lens entropy clusters — but with OPPOSITE SIGNS in the
two prompt styles, replicating across both models:

    gpt2-small   bare     peak +0.717  cluster L6-11   p < 0.0001
    gpt2-small   sentence peak -0.604  cluster L8-11   p < 0.0001
    gpt2-medium  bare     peak +0.983  cluster L12-21  p = 0.0013
    gpt2-medium  sentence peak -0.694  cluster L16-23  p < 0.0001

The pooled effect was smaller than either style's, because the two partly
cancel. A signature of moral polarity should not flip sign with the carrier
sentence, so the leading explanation is that the effect belongs to the FRAME
(what the prompt's final token is, and what prediction problem it poses) rather
than to the moral content of the swapped word.

This corpus holds the frames fixed and replaces the moral contrast with a
non-moral one. Interpretation of the comparison:

  - Neutral pairs show the SAME bare/sentence sign flip, similar magnitude
        -> the effect is a frame artifact; the moral corpus measures syntax,
           not moral polarity, and the moral result should be retired.
  - Neutral pairs show NO comparable effect
        -> something specific to moral contrasts survives, and the moral
           result is worth pursuing.
  - Neutral shows a weaker but same-signed effect
        -> partly frame, partly content; effect sizes must be reported as
           moral-minus-neutral differences, not raw.

--------------------------------------------------------------------------
Design — deliberately parallel to corpus_gen_moral.py
--------------------------------------------------------------------------
5 domains x 2 levels x 2 styles x 3 replicates = 60 pairs / 120 prompts, the
same shape as the moral corpus, so the two are directly comparable and the
same statistics apply with the same power.

The four frames are IDENTICAL to the moral corpus's, including the
"She treated the injured stranger {word}." frame. That frame is semantically
odd with a neutral adverb ("...injured stranger quickly"), and that oddness is
accepted on purpose: changing the frame would confound the control with the
thing it controls for. The frame must be held fixed for the comparison to mean
anything.

Domains are non-moral antonym axes: size, temp (temperature), speed, sound, age.
`role` is "base" for pole A and "contrast" for pole B, matching the moral
corpus's literals so the same pipeline runs unmodified. Pole assignment here is
ARBITRARY — there is no "positive" pole in a neutral contrast, unlike
virtue/vice. Do not read valence into base/contrast for this corpus; the sign of
any difference is only interpretable relative to the moral corpus's sign.

`category` packs the three factors as f"{domain}_{level}_{style}", exactly as
the moral corpus does, recoverable via category.split("_").

--------------------------------------------------------------------------
Token matching
--------------------------------------------------------------------------
Every pair is token-count matched in BOTH the GPT-2 and Pythia tokenizers, by
the same rule and for the same reason as the moral corpus (see its docstring).
The control would be worthless if it reintroduced the length confound the moral
corpus was rebuilt to remove. check_token_matching() enforces this on every run.

Usage:
    python corpus/corpus_gen_neutral.py --output corpus/corpus_neutral.json
    python corpus/corpus_gen_neutral.py --list-categories
"""

import json
import argparse
from pathlib import Path

DOMAIN_NAMES = {
    "size":  "Size",
    "temp":  "Temperature",
    "speed": "Speed",
    "sound": "Sound",
    "age":   "Age",
}

DOMAINS = ["size", "temp", "speed", "sound", "age"]
LEVELS  = ["act", "disposition"]
STYLES  = ["bare", "sentence"]

# Frames are identical to corpus_gen_moral.py's. Do not diverge them.
FRAMES = {
    ("act", "bare"):          "The act of {word} is",
    ("act", "sentence"):      "She treated the injured stranger {word}.",
    ("disposition", "bare"):  "The trait of {word} is",
    ("disposition", "sentence"): "She is a {word} person.",
}

# (domain, level, style) -> (pole A words, pole B words); token-count matched.
WORDS = {
    ("size","act","bare"):        (["expanding","growing","stretching"], ["shrinking","reducing","narrowing"]),
    ("size","act","sentence"):    (["hugely","greatly","widely"],        ["slightly","barely","narrowly"]),
    ("size","disposition","bare"):(["height","length","size"],           ["depth","width","mass"]),
    ("size","disposition","sentence"):(["tall","large","big"],           ["short","small","tiny"]),

    ("temp","act","bare"):        (["heating","warming","boiling"],      ["cooling","chilling","freezing"]),
    ("temp","act","sentence"):    (["warmly","brightly","dryly"],        ["coldly","slowly","dully"]),
    ("temp","disposition","bare"):(["heat","warmth","fire"],             ["cold","chill","ice"]),
    ("temp","disposition","sentence"):(["hot","warm","sunny"],           ["cold","cool","wet"]),

    ("speed","act","bare"):       (["running","rushing","racing"],       ["walking","waiting","resting"]),
    ("speed","act","sentence"):   (["quickly","rapidly","swiftly"],      ["slowly","gradually","steadily"]),
    ("speed","disposition","bare"):(["speed","haste","motion"],          ["delay","rest","calm"]),
    ("speed","disposition","sentence"):(["quick","fast","active"],       ["slow","calm","idle"]),

    ("sound","act","bare"):       (["shouting","yelling","singing"],     ["breathing","sleeping","resting"]),
    ("sound","act","sentence"):   (["loudly","sharply","clearly"],       ["quietly","softly","gently"]),
    ("sound","disposition","bare"):(["noise","sound","volume"],          ["silence","quiet","calm"]),
    ("sound","disposition","sentence"):(["loud","noisy","vocal"],        ["quiet","silent","shy"]),

    ("age","act","bare"):         (["aging","fading","maturing"],        ["restoring","refreshing","renewing"]),
    ("age","act","sentence"):     (["slowly","gradually","steadily"],    ["quickly","suddenly","sharply"]),
    ("age","disposition","bare"): (["age","history","tradition"],        ["youth","novelty","change"]),
    ("age","disposition","sentence"):(["old","ancient","mature"],        ["young","modern","fresh"]),
}

N_REPLICATES = 3
TOKENIZERS_TO_CHECK = ["gpt2", "EleutherAI/pythia-160m"]


def build_pairs():
    """Expand WORDS into a flat list of (base, contrast, category, description)."""
    pairs = []
    for domain in DOMAINS:
        for level in LEVELS:
            for style in STYLES:
                frame = FRAMES[(level, style)]
                a_words, b_words = WORDS[(domain, level, style)]
                category = f"{domain}_{level}_{style}"
                for i in range(N_REPLICATES):
                    description = (
                        f"{DOMAIN_NAMES[domain]}, {level}, {style}, "
                        f"replicate {i + 1}/{N_REPLICATES}"
                    )
                    pairs.append((
                        frame.format(word=a_words[i]),
                        frame.format(word=b_words[i]),
                        category,
                        description,
                    ))
    return pairs


def build_corpus(pairs=None):
    """Convert raw pairs into corpus entries. Schema matches corpus_gen_moral."""
    if pairs is None:
        pairs = build_pairs()
    corpus = []
    for pair_id, (base, contrast, category, description) in enumerate(pairs):
        shared = {"pair_id": pair_id, "category": category, "description": description}
        corpus.append({**shared, "role": "base",     "prompt": base})
        corpus.append({**shared, "role": "contrast", "prompt": contrast})
    return corpus


def check_token_matching(tokenizers=TOKENIZERS_TO_CHECK):
    """Verify both poles of every pair encode to the same token count.

    Matching is required within each tokenizer, not across them. Returns []
    (skip, not pass) when transformers or the tokenizer files are unavailable.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return []
    problems = []
    for hf_name in tokenizers:
        try:
            tok = AutoTokenizer.from_pretrained(hf_name)
        except Exception:
            continue
        for (domain, level, style), (a_words, b_words) in WORDS.items():
            for i, (a, b) in enumerate(zip(a_words, b_words)):
                na = len(tok.encode(" " + a))
                nb = len(tok.encode(" " + b))
                if na != nb:
                    problems.append(
                        f"{domain}/{level}/{style} r{i + 1}: token-count mismatch "
                        f"under {hf_name}: {a!r}={na} vs {b!r}={nb}"
                    )
    return problems


def validate_corpus(pairs=None):
    """Check the expanded pair list. Returns a list of problems; empty is clean."""
    if pairs is None:
        pairs = build_pairs()
    problems = []

    expected = len(DOMAINS) * len(LEVELS) * len(STYLES) * N_REPLICATES
    if len(pairs) != expected:
        problems.append(f"expected {expected} pairs, built {len(pairs)}")

    for domain in DOMAINS:
        for level in LEVELS:
            for style in STYLES:
                key = (domain, level, style)
                if key not in WORDS:
                    problems.append(f"missing words for {key}")
                    continue
                a_words, b_words = WORDS[key]
                for pole, ws in (("A", a_words), ("B", b_words)):
                    if len(ws) != N_REPLICATES:
                        problems.append(
                            f"{key}: pole {pole} has {len(ws)} words, expected {N_REPLICATES}")
                overlap = set(a_words) & set(b_words)
                if overlap:
                    problems.append(f"{key}: word in both poles: {sorted(overlap)}")

    problems.extend(check_token_matching())

    for i, p in enumerate(pairs):
        if p[0] == p[1]:
            problems.append(f"pair {i}: base and contrast are identical: {p[0]!r}")

    seen = set()
    for p in pairs:
        if p[3] in seen:
            problems.append(f"duplicate description: {p[3]!r}")
        seen.add(p[3])

    return problems


def list_categories(corpus):
    categories = {}
    for entry in corpus:
        if entry["role"] != "base":
            continue
        categories.setdefault(entry["category"], []).append(
            f"  [{entry['pair_id']:03d}] {entry['description']}")
    for cat in sorted(categories):
        print(f"\n{cat} ({len(categories[cat])} pairs):")
        for item in categories[cat]:
            print(item)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate the non-moral control corpus for the moral-polarity experiment")
    parser.add_argument("--output", type=str, default="corpus_neutral.json",
                        help="Output file path (default: corpus_neutral.json)")
    parser.add_argument("--list-categories", action="store_true",
                        help="Print corpus categories and exit")
    args = parser.parse_args()

    problems = validate_corpus()
    if problems:
        print("Corpus validation failed:")
        for p in problems:
            print(f"  - {p}")
        return 1

    corpus = build_corpus()

    if args.list_categories:
        list_categories(corpus)
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2)

    n_pairs = len(corpus) // 2
    categories = sorted(set(e["category"] for e in corpus))
    print(f"\nCorpus saved to {output_path}")
    print(f"  {n_pairs} neutral antonym pairs ({len(corpus)} total prompts)")
    print(f"  {len(categories)} categories "
          f"({len(DOMAINS)} domains x {len(LEVELS)} levels x {len(STYLES)} styles)")
    print(f"  Frames identical to corpus_gen_moral.py; pole assignment is arbitrary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
