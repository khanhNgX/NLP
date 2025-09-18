from preprocessing.simple_tokenizer import SimpleTokenizer
from preprocessing.regex_tokenizer import RegexTokenizer
from core.dataset_loaders import load_raw_text_data


if __name__ == "__main__":
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!",
    ]

    print("SimpleTokenizer Results:")
    simple_tokenizer = SimpleTokenizer()
    for s in sentences:
        print(f"Input: {s}")
        print(f"Tokens: {simple_tokenizer.tokenize(s)}\n")

    print("RegexTokenizer Results:")
    regex_tokenizer = RegexTokenizer()
    for s in sentences:
        print(f"Input: {s}")
        print(f"Tokens: {regex_tokenizer.tokenize(s)}\n")

    dataset_path = "Lab1&Lab2/UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)

    sample_text = " ".join(raw_text[:500])
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")