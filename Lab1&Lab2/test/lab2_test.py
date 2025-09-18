import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from representations.count_vectorizer import CountVectorizer
from preprocessing.regex_tokenizer import RegexTokenizer
from core.dataset_loaders import load_raw_text_data

if __name__ == "__main__":
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    vectorizer_corpus = CountVectorizer()
    vectorizer_corpus.fit_transform(corpus)
    print(f"Vocabulary: {vectorizer_corpus.vocabulary_} \n")

    tokenized_corpus = []
    regex_tokenizer = RegexTokenizer()
    for sentence in corpus:
        tokens = regex_tokenizer.tokenize(sentence)
        tokenized_corpus.append(" ".join(tokens))
    print("Tokenized Corpus:")
    for text in tokenized_corpus:    
        print(f"{text}")

    result_corpus = vectorizer_corpus.fit_transform(tokenized_corpus)
    print("\nDocument-Term Matrix:")
    for doc_vector in result_corpus:
        print(doc_vector)
