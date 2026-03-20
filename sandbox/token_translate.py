import argparse
from load_model import model

def token_to_string(token_id):
    """Convert a token ID to its string representation"""
    return model.tokenizer.decode(token_id)

# Usage:
#print(token_to_string(262))      # Should print something like " the"
#print(token_to_string(3797))     # Should print something like " cat"
#print(token_to_string(50256))    # Should print something like "<|endoftext|>"

# Or if you want to batch decode multiple tokens:
def tokens_to_strings(token_ids):
    """Convert list of token IDs to list of strings"""
    return [model.tokenizer.decode(tid) for tid in token_ids]

# Usage:
#tokens = [262, 3797, 2406, 319]
#strings = tokens_to_strings(tokens)
#print(strings)  # ['the', 'cat', 'sat', 'on'] (approximately)


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('token', type=int, nargs=1, default=0, help='input token index')
    return parser.parse_args()


def main():
    args = parse_args()
    print(token_to_string(args.token))

if __name__ == "__main__":
    main()
    

