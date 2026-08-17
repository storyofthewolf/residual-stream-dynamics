"""
corpus_gen.py — Generate a structured prompt corpus with contrast pairs.

Each entry has:
  - pair_id:       shared ID linking a base prompt to its contrast
  - role:          "base" or "contrast"
  - category:      prompt category (pattern, syntactic, predictability,
                   arithmetic, repetition)
  - contrast_type: how the contrast breaks the base (see below)
  - description:   human-readable note on what the pair varies
  - prompt:        the text

The contrast pair design lets the analysis workflows compare internal model
states between prompts that share a stem but diverge in expected predictability.

--------------------------------------------------------------------------
contrast_type — controlling the abstract-noun confound
--------------------------------------------------------------------------
The original n=25 corpus built almost every contrast by appending a low
frequency abstract noun ("philosophy", "democracy", "calculus"). That
confounds two variables: the pattern is broken AND an unusual token appears.
A skeptic can attribute any r_perp effect to lexical frequency rather than to
structure.

This corpus separates them. Every category carries pairs of several types:

  "abstract"   — pattern broken by a low-frequency abstract noun.
                 Matches the original design; kept for continuity.
  "concrete"   — pattern broken by a high-frequency concrete noun
                 ("table", "water"). Same structural break, ordinary token.
  "in_domain"  — pattern broken by a token from the SAME semantic class as
                 the expected continuation ("one two three seven"). The
                 token is unsurprising in isolation; only its position is
                 wrong. This is the tightest control.
  "swap"       — the expected continuation is present but reordered, so
                 token identity is held fixed and only structure changes.

Comparing "abstract" against "in_domain" isolates structure from lexical
frequency: if the r_perp effect survives in "in_domain" pairs, it is not a
frequency artifact.

contrast_type is written to the JSON but is not consumed by extraction.py,
which reads only pair_id / role / category. It is available for stratified
post-hoc analysis via the corpus file.

Usage:
    python corpus/corpus_gen.py --output corpus/base_vs_contrast_n150.json
    python corpus/corpus_gen.py --list-categories
    python corpus/corpus_gen.py --stats
    python corpus/corpus_gen.py --legacy --output corpus/base_vs_contrast_n50.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter

# ============================================================================
# PROMPT PAIRS
# Each entry: (base_prompt, contrast_prompt, category, contrast_type, description)
#
# Base     = higher predictability / more structured
# Contrast = same stem, structure broken
#
# The first entries of each category (marked LEGACY) reproduce the original
# n=25 corpus so results computed before the expansion remain comparable.
# ============================================================================

PROMPT_PAIRS = [

    # ========================================================================
    # PATTERN CONTINUATION
    # Base: strong sequential pattern, model should be confident
    # Contrast: the sequence is interrupted
    # ========================================================================

    # -- LEGACY (original 6) --
    ("one two three four", "one two three purple",
     "pattern", "abstract", "number sequence vs color intrusion"),
    ("Monday Tuesday Wednesday Thursday", "Monday Tuesday Wednesday coffee",
     "pattern", "concrete", "weekday sequence vs noun intrusion"),
    ("January February March April", "January February March democracy",
     "pattern", "abstract", "month sequence vs abstract noun intrusion"),
    ("A B C D", "A B C seven",
     "pattern", "in_domain", "letter sequence vs digit intrusion"),
    ("red blue red blue red", "red blue red blue philosophy",
     "pattern", "abstract", "color alternation vs abstract intrusion"),
    ("10 20 30 40", "10 20 30 banana",
     "pattern", "concrete", "counting by tens vs noun intrusion"),

    # -- in_domain: the intruding token is the right *kind*, wrong position --
    ("one two three four", "one two three seven",
     "pattern", "in_domain", "number sequence vs out-of-order number"),
    ("Monday Tuesday Wednesday Thursday", "Monday Tuesday Wednesday Saturday",
     "pattern", "in_domain", "weekday sequence vs out-of-order weekday"),
    ("January February March April", "January February March October",
     "pattern", "in_domain", "month sequence vs out-of-order month"),
    ("A B C D", "A B C Q",
     "pattern", "in_domain", "letter sequence vs out-of-order letter"),
    ("10 20 30 40", "10 20 30 37",
     "pattern", "in_domain", "counting by tens vs off-pattern number"),
    ("first second third fourth", "first second third ninth",
     "pattern", "in_domain", "ordinal sequence vs out-of-order ordinal"),

    # -- concrete: ordinary high-frequency noun breaks the pattern --
    ("one two three four", "one two three table",
     "pattern", "concrete", "number sequence vs common noun"),
    ("spring summer autumn winter", "spring summer autumn bread",
     "pattern", "concrete", "season sequence vs common noun"),
    ("north south east west", "north south east water",
     "pattern", "concrete", "compass sequence vs common noun"),

    # -- swap: same tokens, order destroyed --
    ("one two three four", "three one four two",
     "pattern", "swap", "number sequence vs shuffled same numbers"),
    ("Monday Tuesday Wednesday Thursday", "Wednesday Monday Thursday Tuesday",
     "pattern", "swap", "weekday sequence vs shuffled same weekdays"),
    ("A B C D", "C A D B",
     "pattern", "swap", "letter sequence vs shuffled same letters"),

    # -- additional abstract, for balance --
    ("spring summer autumn winter", "spring summer autumn philosophy",
     "pattern", "abstract", "season sequence vs abstract noun"),
    ("north south east west", "north south east democracy",
     "pattern", "abstract", "compass sequence vs abstract noun"),
    ("first second third fourth", "first second third calculus",
     "pattern", "abstract", "ordinal sequence vs technical term"),
    ("do re mi fa", "do re mi theorem",
     "pattern", "abstract", "solfege sequence vs mathematical term"),
    ("do re mi fa", "do re mi la",
     "pattern", "in_domain", "solfege sequence vs out-of-order solfege"),
    ("2 4 6 8", "2 4 6 nine",
     "pattern", "in_domain", "even sequence vs odd number"),
    ("2 4 6 8", "2 4 6 elephant",
     "pattern", "concrete", "even sequence vs animal noun"),

    # ========================================================================
    # SYNTACTIC STRUCTURE
    # Base: grammatically well-formed, high completion certainty
    # Contrast: same stem, grammatically or semantically odd
    # ========================================================================

    # -- LEGACY (original 6) --
    ("The cat sat on the", "The cat sat on democracy",
     "syntactic", "abstract", "concrete location vs abstract noun"),
    ("She opened the door and", "She opened the door philosophy",
     "syntactic", "abstract", "coherent continuation stem vs abrupt abstract"),
    ("The dog barked at the", "The dog barked at seventeen",
     "syntactic", "in_domain", "plausible object vs number"),
    ("He picked up the heavy", "He picked up the heavy equation",
     "syntactic", "abstract", "adjective before concrete noun vs abstract noun"),
    ("The sun rises in the", "The sun rises in the algorithm",
     "syntactic", "abstract", "directional completion vs technical noun"),
    ("They walked into the dark", "They walked into the dark calculus",
     "syntactic", "abstract", "adjective before concrete vs discipline name"),

    # -- concrete: the completion is an ordinary noun, just wrong here --
    ("The cat sat on the", "The cat sat on the bread",
     "syntactic", "concrete", "plausible location vs implausible concrete object"),
    ("He picked up the heavy", "He picked up the heavy water",
     "syntactic", "concrete", "adjective before concrete vs mass noun"),
    ("The sun rises in the", "The sun rises in the kitchen",
     "syntactic", "concrete", "directional completion vs concrete place"),
    ("She poured the milk into the", "She poured the milk into the mountain",
     "syntactic", "concrete", "container completion vs landform"),
    ("The bird flew over the", "The bird flew over the spoon",
     "syntactic", "concrete", "landscape completion vs small object"),

    # -- swap: word order violated, vocabulary held fixed --
    ("The cat sat on the mat", "Mat the on sat cat the",
     "syntactic", "swap", "cat-mat sentence vs reversed word order"),
    ("She opened the door slowly", "Slowly door the opened she",
     "syntactic", "swap", "door sentence vs reversed word order"),
    ("The dog barked at the mailman", "Mailman the at barked dog the",
     "syntactic", "swap", "dog-mailman sentence vs reversed word order"),
    ("He read the book quietly", "Quietly book the read he",
     "syntactic", "swap", "book sentence vs reversed word order"),

    # -- in_domain: syntactically valid but semantically anomalous --
    ("The cat sat on the", "The cat sat on the idea",
     "syntactic", "in_domain", "concrete location vs abstract in valid slot"),
    ("They walked into the dark", "They walked into the dark silence",
     "syntactic", "in_domain", "dark-noun completion vs abstract in valid slot"),
    ("The bird flew over the", "The bird flew over the memory",
     "syntactic", "in_domain", "flew-over completion vs abstract in valid slot"),

    # -- additional abstract --
    ("She poured the milk into the", "She poured the milk into the theorem",
     "syntactic", "abstract", "container completion vs mathematical term"),
    ("The bird flew over the", "The bird flew over the philosophy",
     "syntactic", "abstract", "landscape completion vs abstract noun"),
    ("He read the book about the", "He read the book about the calculus",
     "syntactic", "abstract", "topic completion vs technical term"),
    ("The teacher wrote on the", "The teacher wrote on the democracy",
     "syntactic", "abstract", "surface completion vs abstract noun"),
    ("The teacher wrote on the", "The teacher wrote on the river",
     "syntactic", "concrete", "surface completion vs landform"),
    ("He read the book about the", "He read the book about the seventeen",
     "syntactic", "in_domain", "topic completion vs bare number"),

    # ========================================================================
    # HIGH-PREDICTABILITY STEMS
    # Base: cultural / factual completion most readers would agree on
    # Contrast: same stem, unexpected continuation
    # ========================================================================

    # -- LEGACY (original 5) --
    ("To be or not to", "To be or not to dance",
     "predictability", "in_domain", "Shakespeare completion vs verb substitution"),
    ("Once upon a", "Once upon a theorem",
     "predictability", "abstract", "fairy tale opener vs mathematical term"),
    ("In the beginning", "In the beginning calculus",
     "predictability", "abstract", "Genesis opener vs technical term"),
    ("It was a dark and stormy", "It was a dark and stormy equation",
     "predictability", "abstract", "gothic fiction stem vs math term"),
    ("The quick brown fox jumps over the lazy",
     "The quick brown fox jumps over the lazy theorem",
     "predictability", "abstract", "pangram completion vs abstract noun"),

    # -- in_domain: a plausible word class, wrong specific word --
    ("Once upon a", "Once upon a bicycle",
     "predictability", "concrete", "fairy tale opener vs concrete noun"),
    ("It was a dark and stormy", "It was a dark and stormy morning",
     "predictability", "in_domain", "gothic stem vs plausible-but-wrong noun"),
    ("The quick brown fox jumps over the lazy",
     "The quick brown fox jumps over the lazy rabbit",
     "predictability", "in_domain", "pangram completion vs different animal"),
    ("Roses are red violets are", "Roses are red violets are green",
     "predictability", "in_domain", "rhyme completion vs wrong color"),
    ("A picture is worth a thousand", "A picture is worth a thousand rocks",
     "predictability", "concrete", "thousand-words idiom vs concrete noun"),
    ("Better late than", "Better late than sideways",
     "predictability", "in_domain", "idiom completion vs adverb"),
    ("Practice makes", "Practice makes bread",
     "predictability", "concrete", "practice idiom vs concrete noun"),
    ("The early bird catches the", "The early bird catches the algorithm",
     "predictability", "abstract", "idiom completion vs technical term"),
    ("The early bird catches the", "The early bird catches the sandwich",
     "predictability", "concrete", "early-bird idiom vs concrete noun"),
    ("Actions speak louder than", "Actions speak louder than philosophy",
     "predictability", "abstract", "actions idiom vs abstract noun"),
    ("Actions speak louder than", "Actions speak louder than pancakes",
     "predictability", "concrete", "actions idiom vs concrete noun"),
    ("Rome was not built in a", "Rome was not built in a democracy",
     "predictability", "abstract", "Rome idiom vs abstract noun"),
    ("Roses are red violets are", "Roses are red violets are calculus",
     "predictability", "abstract", "rhyme completion vs technical term"),
    ("Practice makes", "Practice makes theorem",
     "predictability", "abstract", "idiom completion vs mathematical term"),
    ("To be or not to", "To be or not to democracy",
     "predictability", "abstract", "Shakespeare completion vs abstract noun"),

    # -- swap --
    ("Once upon a time", "Time a upon once",
     "predictability", "swap", "fairy tale opener vs reversed"),
    ("Better late than never", "Never than late better",
     "predictability", "swap", "idiom vs reversed"),
    ("Practice makes perfect", "Perfect makes practice",
     "predictability", "swap", "idiom vs reordered same words"),

    # ========================================================================
    # ARITHMETIC SURFACE FORM
    # Base: standard arithmetic expression
    # Contrast: same numeric tokens, structure broken
    # (Not testing correctness — studying the forward pass on structured vs
    #  unstructured numeric token sequences.)
    # ========================================================================

    # -- LEGACY (original 5) --
    ("1 + 1 =", "1 + 1 philosophy",
     "arithmetic", "abstract", "arithmetic expression vs noun after operator"),
    ("2 + 2 =", "2 + 2 river",
     "arithmetic", "concrete", "arithmetic expression vs concrete noun"),
    ("3 x 3 =", "3 x 3 democracy",
     "arithmetic", "abstract", "multiplication vs abstract noun"),
    ("10 - 5 =", "10 - 5 Thursday",
     "arithmetic", "in_domain", "subtraction vs weekday"),
    ("5 + 3 =", "5 + 3 elephant",
     "arithmetic", "concrete", "addition vs animal"),

    # -- swap: operator/operand order destroyed, tokens held fixed --
    ("1 + 1 =", "+ 1 = 1",
     "arithmetic", "swap", "expression vs operator-first reordering"),
    ("2 + 2 =", "= 2 2 +",
     "arithmetic", "swap", "expression vs reversed token order"),
    ("3 x 3 =", "x = 3 3",
     "arithmetic", "swap", "multiplication vs scrambled operators"),
    ("10 - 5 =", "- = 10 5",
     "arithmetic", "swap", "subtraction vs scrambled operators"),
    ("7 + 2 =", "+ = 7 2",
     "arithmetic", "swap", "addition vs scrambled operators"),

    # -- in_domain: another number where the answer should go --
    ("1 + 1 =", "1 + 1 7",
     "arithmetic", "in_domain", "1+1 vs bare number replacing ="),
    ("2 + 2 =", "2 + 2 9",
     "arithmetic", "in_domain", "2+2 vs bare number replacing ="),
    ("6 x 7 =", "6 x 7 four",
     "arithmetic", "in_domain", "multiplication vs spelled number"),
    ("12 - 4 =", "12 - 4 nineteen",
     "arithmetic", "in_domain", "subtraction vs spelled number"),
    ("9 + 6 =", "9 + 6 thirty",
     "arithmetic", "in_domain", "addition vs spelled number"),

    # -- concrete / abstract balance --
    ("6 x 7 =", "6 x 7 bread",
     "arithmetic", "concrete", "multiplication vs common noun"),
    ("12 - 4 =", "12 - 4 window",
     "arithmetic", "concrete", "subtraction vs common noun"),
    ("9 + 6 =", "9 + 6 calculus",
     "arithmetic", "abstract", "addition vs technical term"),
    ("7 + 2 =", "7 + 2 theorem",
     "arithmetic", "abstract", "addition vs mathematical term"),
    ("8 / 2 =", "8 / 2 philosophy",
     "arithmetic", "abstract", "division vs abstract noun"),
    ("8 / 2 =", "8 / 2 table",
     "arithmetic", "concrete", "division vs common noun"),

    # ========================================================================
    # REPETITION / LOCAL COHERENCE
    # Base: direct repetition (very low entropy expected)
    # Contrast: repetition broken
    # ========================================================================

    # -- LEGACY (original 3) --
    ("the the the the", "the the the philosophy",
     "repetition", "abstract", "pure repetition vs noun break"),
    ("go go go go", "go go go democracy",
     "repetition", "abstract", "verb repetition vs abstract noun"),
    ("yes yes yes yes", "yes yes yes calculus",
     "repetition", "abstract", "affirmation repetition vs technical term"),

    # -- in_domain: a same-class token breaks the repetition --
    ("the the the the", "the the the a",
     "repetition", "in_domain", "pure repetition vs different article"),
    ("go go go go", "go go go run",
     "repetition", "in_domain", "verb repetition vs different verb"),
    ("yes yes yes yes", "yes yes yes no",
     "repetition", "in_domain", "affirmation repetition vs negation"),
    ("cat cat cat cat", "cat cat cat dog",
     "repetition", "in_domain", "noun repetition vs different animal"),
    ("blue blue blue blue", "blue blue blue green",
     "repetition", "in_domain", "color repetition vs different color"),
    ("one one one one", "one one one two",
     "repetition", "in_domain", "number repetition vs different number"),

    # -- concrete --
    ("cat cat cat cat", "cat cat cat bread",
     "repetition", "concrete", "noun repetition vs unrelated concrete noun"),
    ("blue blue blue blue", "blue blue blue table",
     "repetition", "concrete", "color repetition vs concrete noun"),
    ("one one one one", "one one one water",
     "repetition", "concrete", "number repetition vs mass noun"),
    ("run run run run", "run run run window",
     "repetition", "concrete", "verb repetition vs concrete noun"),

    # -- abstract balance --
    ("cat cat cat cat", "cat cat cat theorem",
     "repetition", "abstract", "noun repetition vs mathematical term"),
    ("run run run run", "run run run democracy",
     "repetition", "abstract", "run repetition vs abstract noun"),
]


# The original n=25-pair corpus, preserved so the pre-expansion results stay
# reproducible. These are exactly the first N entries of each category block
# above that are marked LEGACY.
LEGACY_DESCRIPTIONS = {
    "number sequence vs color intrusion",
    "weekday sequence vs noun intrusion",
    "month sequence vs abstract noun intrusion",
    "letter sequence vs digit intrusion",
    "color alternation vs abstract intrusion",
    "counting by tens vs noun intrusion",
    "concrete location vs abstract noun",
    "coherent continuation stem vs abrupt abstract",
    "plausible object vs number",
    "adjective before concrete noun vs abstract noun",
    "directional completion vs technical noun",
    "adjective before concrete vs discipline name",
    "Shakespeare completion vs verb substitution",
    "fairy tale opener vs mathematical term",
    "Genesis opener vs technical term",
    "gothic fiction stem vs math term",
    "pangram completion vs abstract noun",
    "arithmetic expression vs noun after operator",
    "arithmetic expression vs concrete noun",
    "multiplication vs abstract noun",
    "subtraction vs weekday",
    "addition vs animal",
    "pure repetition vs noun break",
    "verb repetition vs abstract noun",
    "affirmation repetition vs technical term",
}


def legacy_pairs(pairs=PROMPT_PAIRS):
    """Return only the pairs that made up the original n=25 corpus.

    Preserves the original ordering so pair_id values match the pre-expansion
    corpus file exactly.
    """
    seen = set()
    out = []
    for p in pairs:
        desc = p[4]
        if desc in LEGACY_DESCRIPTIONS and desc not in seen:
            seen.add(desc)
            out.append(p)
    return out


def build_corpus(pairs=PROMPT_PAIRS):
    """Convert raw pairs list into structured corpus entries."""
    corpus = []
    for pair_id, (base, contrast, category, contrast_type, description) in enumerate(pairs):
        shared = {
            "pair_id": pair_id,
            "category": category,
            "contrast_type": contrast_type,
            "description": description,
        }
        corpus.append({**shared, "role": "base",     "prompt": base})
        corpus.append({**shared, "role": "contrast", "prompt": contrast})
    return corpus


def validate_corpus(pairs=PROMPT_PAIRS):
    """Check the pair list for the mistakes that are easy to make by hand.

    Returns a list of problem strings; empty means clean.
    """
    problems = []

    # Duplicate descriptions would make legacy_pairs() ambiguous.
    desc_counts = Counter(p[4] for p in pairs)
    for desc, n in desc_counts.items():
        if n > 1:
            problems.append(f"duplicate description ({n}x): {desc!r}")

    # A base/contrast pair that is identical is a copy-paste error.
    for i, p in enumerate(pairs):
        if p[0] == p[1]:
            problems.append(f"pair {i}: base and contrast are identical: {p[0]!r}")

    # Every legacy description must still be present, or the legacy corpus
    # can no longer be regenerated.
    present = {p[4] for p in pairs}
    for desc in LEGACY_DESCRIPTIONS:
        if desc not in present:
            problems.append(f"legacy description missing: {desc!r}")

    valid_types = {"abstract", "concrete", "in_domain", "swap"}
    for i, p in enumerate(pairs):
        if p[3] not in valid_types:
            problems.append(f"pair {i}: unknown contrast_type {p[3]!r}")

    return problems


def list_categories(corpus):
    categories = {}
    for entry in corpus:
        if entry["role"] != "base":
            continue
        cat = entry["category"]
        categories.setdefault(cat, [])
        categories[cat].append(
            f"  [{entry['pair_id']:03d}] ({entry['contrast_type']:9s}) {entry['description']}"
        )
    for cat in sorted(categories):
        items = categories[cat]
        print(f"\n{cat} ({len(items)} pairs):")
        for item in items:
            print(item)
    print()


def print_stats(corpus):
    """Print the category x contrast_type design matrix."""
    bases = [e for e in corpus if e["role"] == "base"]
    cats  = sorted({e["category"] for e in bases})
    types = ["abstract", "concrete", "in_domain", "swap"]

    print(f"\n{len(bases)} pairs ({len(corpus)} prompts)\n")
    header = f"{'category':16s}" + "".join(f"{t:>11s}" for t in types) + f"{'total':>8s}"
    print(header)
    print("-" * len(header))
    for cat in cats:
        row = [e for e in bases if e["category"] == cat]
        counts = [sum(1 for e in row if e["contrast_type"] == t) for t in types]
        print(f"{cat:16s}" + "".join(f"{c:>11d}" for c in counts) + f"{len(row):>8d}")
    print("-" * len(header))
    totals = [sum(1 for e in bases if e["contrast_type"] == t) for t in types]
    print(f"{'total':16s}" + "".join(f"{c:>11d}" for c in totals) + f"{len(bases):>8d}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate prompt corpus with contrast pairs")
    parser.add_argument("--output", type=str, default="corpus.json",
                        help="Output file path (default: corpus.json)")
    parser.add_argument("--list-categories", action="store_true",
                        help="Print corpus categories and exit")
    parser.add_argument("--stats", action="store_true",
                        help="Print the category x contrast_type design matrix and exit")
    parser.add_argument("--legacy", action="store_true",
                        help="Emit only the original 25 pairs (pre-expansion corpus)")
    args = parser.parse_args()

    problems = validate_corpus()
    if problems:
        print("Corpus validation failed:")
        for p in problems:
            print(f"  - {p}")
        return 1

    pairs  = legacy_pairs() if args.legacy else PROMPT_PAIRS
    corpus = build_corpus(pairs)

    if args.list_categories:
        list_categories(corpus)
        return 0

    if args.stats:
        print_stats(corpus)
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2)

    n_pairs = len(corpus) // 2
    categories = sorted(set(e["category"] for e in corpus))
    print(f"\nCorpus saved to {output_path}")
    print(f"  {n_pairs} contrast pairs ({len(corpus)} total prompts)")
    print(f"  Categories: {', '.join(categories)}")
    print(f"\nRun with --stats to see the category x contrast_type matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
