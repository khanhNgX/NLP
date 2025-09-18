import re
from core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())
