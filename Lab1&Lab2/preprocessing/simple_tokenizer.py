from core.interfaces import Tokenizer
import string

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        for c in text:
            if c in string.punctuation:
                text = text.replace(c, f' {c} ')
        tokens = text.split()
        return tokens
