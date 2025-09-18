from core.interfaces import Vectorizer
from preprocessing.regex_tokenizer import RegexTokenizer


class CountVectorizer(Vectorizer):
    def __init__(self):
        self.vocabulary_ = {}
    
    def fit(self, corpus: list[str]):
        unique_tokens = set()
        regex_tokenizer = RegexTokenizer()
        for document in corpus:
            tokens = regex_tokenizer.tokenize(document)
            unique_tokens.update(tokens)
        
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
    
    def transform(self, documents: list[str]) -> list[list[int]]:
        if not self.vocabulary_:
            raise ValueError("The vectorizer has not been fitted yet.")
        
        vectors = []
        for document in documents:
            tokens = document.split()
            vector = [0] * len(self.vocabulary_)
            for token in tokens:
                if token in self.vocabulary_:
                    index = self.vocabulary_[token]
                    vector[index] += 1
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)