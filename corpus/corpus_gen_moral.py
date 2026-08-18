"""
corpus_gen_moral.py — Generate a moral-polarity prompt corpus with virtue/vice pairs.

This is an INDEPENDENT corpus from corpus_gen.py, testing a different axis.
corpus_gen.py tests *predictability* (a coherent prompt vs. one that breaks
pattern or syntax). This file tests *moral polarity*: a virtue-pole prompt vs.
a vice-pole prompt drawn from Moral Foundations Theory. The two corpora share a
JSON schema and a downstream pipeline, but nothing else. Do not merge or
conflate their semantics.

--------------------------------------------------------------------------
role encodes moral polarity here, NOT predictability
--------------------------------------------------------------------------
`role` takes the literal values "base" (virtue pole) and "contrast" (vice pole).
These strings are reused deliberately: entropy_plots.py and entropy_compute.py
contain hardcoded checks against "base"/"contrast", so reusing the literals lets
plot_paired_difference() and print_summary() work against this corpus with zero
modification. Read "base" as "virtue" and "contrast" as "vice" throughout.

--------------------------------------------------------------------------
category packs three factors into one string
--------------------------------------------------------------------------
The design is 5 foundations x 2 levels x 2 styles x 3 replicates = 60 pairs
(120 prompts). ActivationRecord and EntropyRecord carry only pair_id / role /
category, with no field for a third factor, so the three sub-factors are packed
into the category string as f"{foundation}_{level}_{style}", e.g. "care_act_bare"
or "fairness_disposition_sentence". Downstream analysis recovers them with
category.split("_"). This avoids a dataclass and .npz schema change.

  foundations: care, fairness, loyalty, authority, sanctity
  levels:      act, disposition
  styles:      bare, sentence

`description` is human-readable only ("Care/Harm, act, bare, replicate 1/3").
extraction.py reads only pair_id / role / category from a corpus entry, so
description never reaches a Record; it exists for --list-categories output and
corpus-file readability.

--------------------------------------------------------------------------
Caveat: the Authority/Subversion foundation is not symmetric with the others
--------------------------------------------------------------------------
This foundation's vice pole maps defiance -> negative, following the
conventional operationalization in standard MFT dictionaries. Haidt frames the
foundation as respect for *legitimate* hierarchy specifically, and defiance of
illegitimate authority is usually judged virtuous. We follow the conventional
operationalization here, but this asymmetry should be flagged as a caveat in any
writeup rather than treated as equivalent to the other four foundations.

--------------------------------------------------------------------------
NOTE — intensity is NOT implemented in this pass
--------------------------------------------------------------------------
The three replicates per cell are informal mild/moderate/severe orderings by
general usage judgment. They are NOT validated against any lexicon and carry no
intensity semantics in the emitted corpus. A future pass will re-derive both the
word choices and their ordering from continuous scores in MoralStrength (Araque
et al. 2020) or eMFD (Hopp et al. 2021). That pass will require a new
`intensity_score` field threaded through ActivationRecord and EntropyRecord and
through their .npz save/load functions — a real schema change, explicitly out of
scope now.

Do NOT encode intensity into `category` or `pair_id`. Doing so would fragment
the 3 replicates into separate categories and defeat their purpose: they are
pooled at N=3 per cell to give bootstrap CI headroom later.

Usage:
    python corpus/corpus_gen_moral.py --output corpus/corpus_moral.json
    python corpus/corpus_gen_moral.py --list-categories
"""

import json
import argparse
from pathlib import Path

# ============================================================================
# TEMPLATES
#
# Keyed by (foundation, level, style). Each value is:
#     (template, [3 virtue words], [3 vice words])
#
# The template is fixed within a cell; only the bracketed word swaps between the
# virtue and vice poles and across the 3 replicates. Replicate i pairs
# virtue[i] against vice[i], so base and contrast differ in exactly one word.
# ============================================================================

FOUNDATION_NAMES = {
    "care":      "Care/Harm",
    "fairness":  "Fairness/Cheating",
    "loyalty":   "Loyalty/Betrayal",
    "authority": "Authority/Subversion",
    "sanctity":  "Sanctity/Degradation",
}

FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "sanctity"]
LEVELS      = ["act", "disposition"]
STYLES      = ["bare", "sentence"]

TEMPLATES = {
    # ---- Care/Harm ----
    ("care", "act", "bare"): (
        "The act of {word} is",
        ["comforting", "protecting", "nurturing"],
        ["tormenting", "neglecting", "harming"],
    ),
    ("care", "act", "sentence"): (
        "She treated the injured stranger {word}.",
        ["gently", "kindly", "compassionately"],
        ["cruelly", "harshly", "brutally"],
    ),
    ("care", "disposition", "bare"): (
        "The trait of {word} is",
        ["kindness", "compassion", "gentleness"],
        ["cruelty", "harshness", "brutality"],
    ),
    ("care", "disposition", "sentence"): (
        "She is a {word} person.",
        ["kind", "gentle", "compassionate"],
        ["cruel", "harsh", "brutal"],
    ),

    # ---- Fairness/Cheating ----
    ("fairness", "act", "bare"): (
        "The act of {word} is",
        ["sharing", "cooperating", "reciprocating"],
        ["cheating", "exploiting", "stealing"],
    ),
    ("fairness", "act", "sentence"): (
        "She treated her business partner {word}.",
        ["fairly", "honestly", "equitably"],
        ["unfairly", "dishonestly", "deceitfully"],
    ),
    ("fairness", "disposition", "bare"): (
        "The trait of {word} is",
        ["fairness", "honesty", "integrity"],
        ["dishonesty", "corruption", "deceit"],
    ),
    ("fairness", "disposition", "sentence"): (
        "She is a {word} person.",
        ["fair", "honest", "principled"],
        ["dishonest", "corrupt", "deceitful"],
    ),

    # ---- Loyalty/Betrayal ----
    ("loyalty", "act", "bare"): (
        "The act of {word} is",
        ["supporting", "defending", "upholding"],
        ["betraying", "abandoning", "deserting"],
    ),
    ("loyalty", "act", "sentence"): (
        "She treated her closest friend {word}.",
        ["loyally", "faithfully", "devotedly"],
        ["disloyally", "treacherously", "unfaithfully"],
    ),
    ("loyalty", "disposition", "bare"): (
        "The trait of {word} is",
        ["loyalty", "faithfulness", "devotion"],
        ["disloyalty", "treachery", "betrayal"],
    ),
    ("loyalty", "disposition", "sentence"): (
        "She is a {word} person.",
        ["loyal", "faithful", "devoted"],
        ["disloyal", "treacherous", "unfaithful"],
    ),

    # ---- Authority/Subversion ----
    # See the module docstring: this foundation's poles are not symmetric with
    # the other four. Conventional MFT operationalization is followed here.
    ("authority", "act", "bare"): (
        "The act of {word} is",
        ["obeying", "deferring", "complying"],
        ["defying", "disobeying", "rebelling"],
    ),
    ("authority", "act", "sentence"): (
        "She treated her commanding officer {word}.",
        ["obediently", "deferentially", "dutifully"],
        ["defiantly", "insubordinately", "rebelliously"],
    ),
    ("authority", "disposition", "bare"): (
        "The trait of {word} is",
        ["obedience", "deference", "discipline"],
        ["insubordination", "defiance", "rebelliousness"],
    ),
    ("authority", "disposition", "sentence"): (
        "She is a {word} person.",
        ["obedient", "dutiful", "disciplined"],
        ["insubordinate", "defiant", "rebellious"],
    ),

    # ---- Sanctity/Degradation ----
    ("sanctity", "act", "bare"): (
        "The act of {word} is",
        ["purifying", "honoring", "revering"],
        ["defiling", "desecrating", "contaminating"],
    ),
    ("sanctity", "act", "sentence"): (
        "She treated the sacred site {word}.",
        ["reverently", "respectfully", "devoutly"],
        ["disrespectfully", "sacrilegiously", "profanely"],
    ),
    ("sanctity", "disposition", "bare"): (
        "The trait of {word} is",
        ["purity", "reverence", "sanctity"],
        ["impurity", "profanity", "depravity"],
    ),
    ("sanctity", "disposition", "sentence"): (
        "She is a {word} person.",
        ["pure", "reverent", "devout"],
        ["impure", "profane", "depraved"],
    ),
}

N_REPLICATES = 3


def build_pairs():
    """Expand TEMPLATES into a flat list of raw pairs.

    Each entry: (base_prompt, contrast_prompt, category, description),
    where base is the virtue pole and contrast is the vice pole.
    Iteration order is foundation -> level -> style -> replicate, so pair_id
    assignment in build_corpus() is deterministic.
    """
    pairs = []
    for foundation in FOUNDATIONS:
        for level in LEVELS:
            for style in STYLES:
                template, virtues, vices = TEMPLATES[(foundation, level, style)]
                category = f"{foundation}_{level}_{style}"
                for i in range(N_REPLICATES):
                    description = (
                        f"{FOUNDATION_NAMES[foundation]}, {level}, {style}, "
                        f"replicate {i + 1}/{N_REPLICATES}"
                    )
                    pairs.append((
                        template.format(word=virtues[i]),
                        template.format(word=vices[i]),
                        category,
                        description,
                    ))
    return pairs


def build_corpus(pairs=None):
    """Convert the raw pairs list into structured corpus entries.

    Emits the same flat schema as corpus_gen.build_corpus(): pair_id, role,
    category, description, prompt. `role` is "base" for the virtue pole and
    "contrast" for the vice pole (see module docstring).
    """
    if pairs is None:
        pairs = build_pairs()
    corpus = []
    for pair_id, (base, contrast, category, description) in enumerate(pairs):
        shared = {
            "pair_id": pair_id,
            "category": category,
            "description": description,
        }
        corpus.append({**shared, "role": "base",     "prompt": base})
        corpus.append({**shared, "role": "contrast", "prompt": contrast})
    return corpus


def validate_corpus(pairs=None):
    """Check the expanded pair list for the mistakes that are easy to make.

    Returns a list of problem strings; empty means clean.
    """
    if pairs is None:
        pairs = build_pairs()
    problems = []

    expected = len(FOUNDATIONS) * len(LEVELS) * len(STYLES) * N_REPLICATES
    if len(pairs) != expected:
        problems.append(f"expected {expected} pairs, built {len(pairs)}")

    # Every cell must exist and carry exactly N_REPLICATES words per pole.
    for foundation in FOUNDATIONS:
        for level in LEVELS:
            for style in STYLES:
                key = (foundation, level, style)
                if key not in TEMPLATES:
                    problems.append(f"missing template for {key}")
                    continue
                template, virtues, vices = TEMPLATES[key]
                if "{word}" not in template:
                    problems.append(f"{key}: template has no {{word}} slot")
                for pole, words in (("virtue", virtues), ("vice", vices)):
                    if len(words) != N_REPLICATES:
                        problems.append(
                            f"{key}: {pole} has {len(words)} words, "
                            f"expected {N_REPLICATES}"
                        )
                overlap = set(virtues) & set(vices)
                if overlap:
                    problems.append(f"{key}: word in both poles: {sorted(overlap)}")

    # An identical base/contrast pair is a copy-paste error.
    for i, p in enumerate(pairs):
        if p[0] == p[1]:
            problems.append(f"pair {i}: base and contrast are identical: {p[0]!r}")

    # Descriptions must be unique so --list-categories is unambiguous.
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
        cat = entry["category"]
        categories.setdefault(cat, [])
        categories[cat].append(
            f"  [{entry['pair_id']:03d}] {entry['description']}"
        )
    for cat in sorted(categories):
        items = categories[cat]
        print(f"\n{cat} ({len(items)} pairs):")
        for item in items:
            print(item)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate moral-polarity prompt corpus with virtue/vice pairs")
    parser.add_argument("--output", type=str, default="corpus_moral.json",
                        help="Output file path (default: corpus_moral.json)")
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
    print(f"  {n_pairs} virtue/vice pairs ({len(corpus)} total prompts)")
    print(f"  {len(categories)} categories "
          f"({len(FOUNDATIONS)} foundations x {len(LEVELS)} levels x {len(STYLES)} styles)")
    print(f"  role: 'base' = virtue pole, 'contrast' = vice pole")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
