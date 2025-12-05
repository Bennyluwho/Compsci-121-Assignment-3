import json
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

class SearchEngine:
    def __init__(self, index_path: str, docids_path: str):
        # load the entire index into memory for fast lookups
        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(docids_path, "r", encoding="utf-8") as f:
            self.docids = json.load(f)
        
        self.total_docs = len(self.docids)
    
    def _compute_idf(self, token: str) -> float:
        # log(total_docs / doc_frequency)
        if token not in self.index:
            return 0.0
        df = len(self.index[token])
        if df == 0:
            return 0.0
        return math.log(self.total_docs / df)
    
    def _compute_tf_idf(self, token: str, doc_id: int, tf: int, imp: int = 0) -> float:
        # tf-idf score + important words
        idf = self._compute_idf(token)
        base_score = tf * idf
        important_boost = imp * idf * 1.5  # 1.5 for important words
        return base_score + important_boost
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        # tokenize and stem the query (same process as indexing)
        query_tokens = tokenize_and_stem(query)
        if not query_tokens:
            return []
        
        doc_scores = defaultdict(float)
        matched_tokens = 0
        
        # accumulate scores for each document containing query terms
        for token in query_tokens:
            if token not in self.index:
                continue
            matched_tokens += 1
            
            idf = self._compute_idf(token)
            # iterate through all postings for this token
            for posting in self.index[token]:
                doc_id = posting["doc_id"]
                tf = posting["tf"]
                imp = posting.get("imp", 0)
                tf_idf = self._compute_tf_idf(token, doc_id, tf, imp)
                doc_scores[doc_id] += tf_idf  # sum scores across query terms
        
        if matched_tokens == 0:
            return []
        
        # sort by score and return top k results
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
    # initialize search engine with pre-built index
    engine = SearchEngine("final_index.json", "final_docids.json")
    
    # interactive search loop
    while True:
        query = input(">>> ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        
        # measure query response time
        start_time = time.time()
        results = engine.search(query, top_k=10)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        if results:
            print(f"found {len(results)} reports in {response_time_ms:.2f} milliseconds")
            for i, (url, score) in enumerate(results, 1):
                print(f"{i}. {url} (score: {score:.4f})")
        else:
            print("No results found.")
        print()
