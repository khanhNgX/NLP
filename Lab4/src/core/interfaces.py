from abc import abstractmethod
from typing import List

class Tokenizer():
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass
