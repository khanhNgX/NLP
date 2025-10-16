from src.core.interfaces import Tokenizer
import string

class SimpleTokenizer(Tokenizer):
    """
    Chuong trinh tach tu don gian.
    Chuyen van ban thanh chu thuong, them khoang trang xung quanh dau cau, roi tach tu.
    """
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        for c in text:
            if c in string.punctuation:
                text = text.replace(c, f' {c} ')
        tokens = text.split()
        return tokens
