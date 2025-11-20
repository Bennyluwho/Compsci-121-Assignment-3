import json
from collections import defaultdict
from typing import List
class IndexMerger:
    def __init__(self):
        #token -> {doc_id -> tf}
        self.merged_index: dict[str, dict[int, int]] = defaultdict(dict)
        #doc_id -> url
        self.merged_docids: dict[int,str] = {}
    
    def merge_partial_indexes(self, partial_index_paths: list[str]):
        #Reads each partial index JSON and merges into a single in memory index
        #For each token it combines postings from all partials and accumulates frequencies if the same doc_id appears multiple times.
        self.merged_index = defaultdict(dict)

        for path in partial_index_paths:
            with open(path, "r", encoding="utf-8") as f:
                partial = json.load(f)
            
            for token, postings in partial.items():
                posting_map = self.merged_index[token]

                for p in postings:
                    doc_id = p["doc_id"]
                    tf = p["tf"]

                    if doc_id in posting_map:
                        posting_map[doc_id] += tf
                    else: 
                        posting_map[doc_id] = tf

    def merge_partial_docids(self, partial_docids_paths: List[str]):
        #Merges all partial docid mapping files into one dictionary
        #Since doc_ids are global and unique, this is just a union
        self.merged_docids = {}

        for path in partial_docids_paths:
            with open(path, "r", encoding="utf-8") as f:
                partial = json.load(f)

            for doc_id_str, url in partial.items():
                doc_id = int(doc_id_str)
                self.merged_docids[doc_id] = url

    def save_final_index(self, index_path):
        #saves final merged index to a json file
        serializable_index: dict[str, list[dict[str,int]]] = {}

        for token, posting_map in self.merged_index.items():
            postings = [
                {"doc_id": doc_id, "tf": tf}
                for doc_id, tf in posting_map.items()
            ]
            postings.sort(key=lambda p: p["doc_id"])
            serializable_index[token] = postings
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(serializable_index, f)
    
    def save_final_docids(self, docid_path: str):
        #saves the merged docid -> url mappig to JSON.
        #Keys will be strings due to JSON, but can be loaded back into ints when loading.
        serializable_docids = {
            str(doc_id) : url for doc_id, url in self.merged_docids.items()
        }

        with open(docid_path, "w", encoding="utf-8") as f:
            json.dump(serializable_docids, f)
    