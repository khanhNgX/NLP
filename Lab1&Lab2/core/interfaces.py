from abc import abstractmethod
from typing import List

class Tokenizer():
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

class Vectorizer():
    @abstractmethod
    def fit(self, corpus: list[str]):
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass

    @abstractmethod
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        pass