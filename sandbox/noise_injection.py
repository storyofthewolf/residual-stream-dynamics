import random
import torch
from load_model import model, device

def add_typos(prompt, noise_level):
    """Scramble letters in prompt"""
    # noise_level = 0.0 (clean) to 1.0 (gibberish)
    chars = list(prompt)
    num_scramble = int(len(chars) * noise_level)
    indices = random.sample(range(len(chars)), num_scramble)
    for i in indices:
        chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz ')
    return ''.join(chars)


def delete_words(prompt, noise_level):
    """Remove random words"""
    words = prompt.split()
    num_delete = int(len(words) * noise_level)
    indices = random.sample(range(len(words)), num_delete)
    remaining = [w for i, w in enumerate(words) if i not in indices]
    return ' '.join(remaining)


def main():
    lyric = "Over thinking, over analyzing separates the body from the mind, Withering my intuition, missing opportunities and I must, Feed my will to feel my moment, drawing way outside the lines"
    
    prompts = [
        (lyric, 0.01),
        (lyric, 0.05),
        (lyric, 0.1),
        (lyric, 0.15),
        (lyric, 0.2),
        (lyric, 0.25),

    ]

    for prompt in prompts:
        newprompt = add_noise(prompt[0], prompt[1])
        print(prompt[1], newprompt)
    
if __name__ == "__main__":
    main()


"""

This creates **controlled degradation**. Much more rigorous than just testing "gibberish."

**Different models:**
- GPT-2 Small (already works)
- GPT-2 Medium (if it fits)
- Smaller open models (Llama 1B, Phi, etc.)

If the pattern holds across models, your claim is stronger.

---

## The Real Innovation

What you should actually test:
```
Hypothesis 1: Clear input → fast entropy drop (you've shown this)

Hypothesis 2: Entropy drop timing correlates with 
              when interpretable features emerge (SAE angle)

Hypothesis 3: High-entropy layers make worse predictions
              for corrupted inputs (downstream effect)

Hypothesis 4: Effect size correlates with model size
              (bigger models compress faster?)
"""
