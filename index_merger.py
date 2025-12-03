import json
from collections import defaultdict

class IndexMerger:
    def __init__(self):
        self.merged_index = defaultdict(dict)
        self.merged_docids = {}

    def merge_partial_indexes(self, partial_index_paths: list[str]):
        # merge postings with same doc_id
        for index_path in partial_index_paths:
            with open(index_path, "r", encoding="utf-8") as f:
                partial_index = json.load(f)
            
            for token, postings in partial_index.items():
                for posting in postings:
                    doc_id = posting["doc_id"]
                    tf = posting["tf"]
                    imp = posting.get("imp", 0)
                    
                    if doc_id in self.merged_index[token]:
                        self.merged_index[token][doc_id] = (self.merged_index[token][doc_id][0] + tf, 
                                                             self.merged_index[token][doc_id][1] + imp)
                    else:
                        self.merged_index[token][doc_id] = (tf, imp)

    # merge all docids
    def merge_partial_docids(self, partial_docid_paths: list[str]):
        for docid_path in partial_docid_paths:
            with open(docid_path, "r", encoding="utf-8") as f:
                partial_docids = json.load(f)
                self.merged_docids.update(partial_docids)

    def save_final_index(self, index_path):
        serializable_index = {}
        for token, doc_dict in self.merged_index.items():
            postings_list = [{"doc_id": doc_id, "tf": tf, "imp": imp} 
                           for doc_id, (tf, imp) in doc_dict.items()]
            serializable_index[token] = postings_list
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(serializable_index, f)
    
    def split_index_by_term_ranges(self, index_path: str, num_splits: int = 10):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        sorted_tokens = sorted(index.keys())
        tokens_per_split = len(sorted_tokens) // num_splits
        
        split_files = []
        for i in range(num_splits):
            start_idx = i * tokens_per_split
            if i == num_splits - 1:
                end_idx = len(sorted_tokens)
            else:
                end_idx = (i + 1) * tokens_per_split
            
            split_tokens = sorted_tokens[start_idx:end_idx]
            split_index = {token: index[token] for token in split_tokens}
            
            split_path = f"index_split_{i:02d}.json"
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(split_index, f)
            
            split_files.append({
                "path": split_path,
                "start_token": split_tokens[0] if split_tokens else "",
                "end_token": split_tokens[-1] if split_tokens else ""
            })
            
            print(f"Created split {i}: {split_path} ({len(split_tokens)} tokens)")
        
        # Tell main where and how to merge
        manifest_path = "index_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(split_files, f, indent=2)
        
        print(f"Index manifest saved to {manifest_path}")
        return manifest_path
    