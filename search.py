from bs4 import BeautifulSoup
import json
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from collections import defaultdict
import time
from postings import Posting

tokenizer = RegexpTokenizer(r"[a-zA-Z0-9\-]+")
stemmer = SnowballStemmer("english") # swtiched to a faster stemmer

def tokenize_and_stem(text: str) -> list[str]:
        text = text.lower()
        tokens = tokenizer.tokenize(text)
        tokens = [t for t in tokens] 
        stems = [stemmer.stem(t) for t in tokens]
        return stems

def search_tokens(tokens, index):
        results = []
        result = {}

        for token in tokens: 
                if token in index:
                        result[token] = index[token]
                else:
                        result[token] = []

                results.append(result) 
        return result

if __name__ == "__main__":
    with open("partial_index_0.json", "r", encoding="utf-8") as f:
        index = json.load(f)

    q = input(">>>")
    t = tokenize_and_stem(q)
    res = search_tokens(t, index)
    print(t)
    print(res)

