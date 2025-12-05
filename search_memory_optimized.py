import json
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
import time

tokenizer = RegexpTokenizer(r"[a-zA-Z0-9\-]+")
stemmer = PorterStemmer()

def tokenize_and_stem(text: str) -> list[str]:
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    return stems

class MemoryOptimizedSearchEngine:
    def __init__(self, manifest_path: str, docids_path: str):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        
        with open(docids_path, "r", encoding="utf-8") as f:
            self.docids = json.load(f)
        
        self.total_docs = len(self.docids)
        self.index_cache = {}
        self.split_cache = {}
        self.doc_lengths_cache = {}
    
    def _find_split_for_token(self, token: str) -> str:
        for split_info in self.manifest:
            start = split_info["start_token"]
            end = split_info["end_token"]
            if start <= token <= end:
                return split_info["path"]
        return None
    
    def _load_token_postings(self, token: str):
        if token in self.index_cache:
            return self.index_cache[token]
        
        split_path = self._find_split_for_token(token)
        if not split_path:
            return []
        
        if split_path not in self.split_cache:
            with open(split_path, "r", encoding="utf-8") as f:
                self.split_cache[split_path] = json.load(f)
        
        split_index = self.split_cache[split_path]
        postings = split_index.get(token, [])
        self.index_cache[token] = postings
        return postings
    
    
    def _compute_idf(self, token: str) -> float:
        postings = self._load_token_postings(token)
        if not postings:
            return 0.0
        
        df = len(postings)
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
            postings = self._load_token_postings(token)
            if not postings:
                continue
            
            matched_tokens += 1
            idf = self._compute_idf(token)
            
            for posting in postings:
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

if __name__ == "__main__":
    engine = MemoryOptimizedSearchEngine("index_manifest.json", "final_docids.json")
    
    while True:
        query = input(">>> ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        
        start_time = time.time()
        results = engine.search(query, top_k=10)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, (url, score) in enumerate(results, 1):
                print(f"{i}. {url} (score: {score:.4f})")
            print(f"found {len(results)} reports in {response_time_ms:.2f} milliseconds")
        else:
            print("No results found.")
        print()

