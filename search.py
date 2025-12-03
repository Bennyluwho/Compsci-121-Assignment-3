import json
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from collections import defaultdict
import math

tokenizer = RegexpTokenizer(r"[a-zA-Z0-9\-]+")
stemmer = PorterStemmer()

def tokenize_and_stem(text: str) -> list[str]:
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    return stems

class SearchEngine:
    def __init__(self, index_path: str, docids_path: str):
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(docids_path, "r", encoding="utf-8") as f:
            self.docids = json.load(f)
        
        self.total_docs = len(self.docids)
        self.doc_lengths = {}
        self._compute_doc_lengths()
    
    def _compute_doc_lengths(self):
        for token, postings in self.index.items():
            for posting in postings:
                doc_id = posting["doc_id"]
                tf = posting["tf"]
                if doc_id not in self.doc_lengths:
                    self.doc_lengths[doc_id] = 0
                self.doc_lengths[doc_id] += tf
    
    def _compute_idf(self, token: str) -> float:
        if token not in self.index:
            return 0.0
        df = len(self.index[token])
        if df == 0:
            return 0.0
        return math.log(self.total_docs / df)
    
    def _compute_tf_idf(self, token: str, doc_id: int, tf: int, imp: int = 0) -> float:
        idf = self._compute_idf(token)
        base_score = tf * idf
        important_boost = imp * idf * 1.5
        return base_score + important_boost
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_tokens = tokenize_and_stem(query)
        if not query_tokens:
            return []
        
        doc_scores = defaultdict(float)
        matched_tokens = 0
        
        for token in query_tokens:
            if token not in self.index:
                continue
            matched_tokens += 1
            
            idf = self._compute_idf(token)
            for posting in self.index[token]:
                doc_id = posting["doc_id"]
                tf = posting["tf"]
                imp = posting.get("imp", 0)
                tf_idf = self._compute_tf_idf(token, doc_id, tf, imp)
                doc_scores[doc_id] += tf_idf
        
        if matched_tokens == 0:
            return []
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc_id_str = str(doc_id)
            if doc_id_str in self.docids:
                url = self.docids[doc_id_str]
                results.append((url, score))
        
        return results
    
    def search_and(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_tokens = tokenize_and_stem(query)
        if not query_tokens:
            return []
        
        doc_sets = []
        for token in query_tokens:
            if token in self.index:
                doc_set = {posting["doc_id"] for posting in self.index[token]}
                doc_sets.append(doc_set)
            else:
                return []
        
        if not doc_sets:
            return []
        
        common_docs = doc_sets[0]
        for doc_set in doc_sets[1:]:
            common_docs = common_docs.intersection(doc_set)
        
        doc_scores = defaultdict(float)
        for doc_id in common_docs:
            for token in query_tokens:
                if token in self.index:
                    for posting in self.index[token]:
                        if posting["doc_id"] == doc_id:
                            tf = posting["tf"]
                            imp = posting.get("imp", 0)
                            tf_idf = self._compute_tf_idf(token, doc_id, tf, imp)
                            doc_scores[doc_id] += tf_idf
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc_id_str = str(doc_id)
            if doc_id_str in self.docids:
                url = self.docids[doc_id_str]
                results.append((url, score))
        
        return results

if __name__ == "__main__":
    engine = SearchEngine("final_index.json", "final_docids.json")
    
    while True:
        query = input(">>> ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        
        results = engine.search(query, top_k=10)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, (url, score) in enumerate(results, 1):
                print(f"{i}. {url} (score: {score:.4f})")
        else:
            print("No results found.")
        print()
