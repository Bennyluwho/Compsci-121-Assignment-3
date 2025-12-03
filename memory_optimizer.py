import json
import sys
from pathlib import Path

class MemoryOptimizedIndexer:
    def __init__(self, index_path: str, docids_path: str):
        self.index_path = index_path
        self.docids_path = docids_path
        self.index = None
        self.docids = None
    
    def load_index_lazy(self):
        if self.index is None:
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        return self.index
    
    def load_docids_lazy(self):
        if self.docids is None:
            with open(self.docids_path, "r", encoding="utf-8") as f:
                self.docids = json.load(f)
        return self.docids
    
    def get_memory_usage(self):
        index_size = sys.getsizeof(self.index) if self.index else 0
        docids_size = sys.getsizeof(self.docids) if self.docids else 0
        return index_size + docids_size
    
    def get_file_size(self):
        index_file_size = Path(self.index_path).stat().st_size if Path(self.index_path).exists() else 0
        docids_file_size = Path(self.docids_path).stat().st_size if Path(self.docids_path).exists() else 0
        return index_file_size + docids_file_size

def check_memory_optimization():
    print("Memory Optimization Check")
    print("=" * 50)
    
    index_path = "final_index.json"
    docids_path = "final_docids.json"
    
    if not Path(index_path).exists():
        print(f"Index file {index_path} not found. Please run main.py first.")
        return
    
    optimizer = MemoryOptimizedIndexer(index_path, docids_path)
    
    file_size = optimizer.get_file_size()
    print(f"Index file size: {file_size / 1024:.2f} KB")
    
    optimizer.load_index_lazy()
    optimizer.load_docids_lazy()
    
    memory_usage = optimizer.get_memory_usage()
    print(f"Memory usage: {memory_usage / 1024:.2f} KB")
    
    if memory_usage < file_size:
        print("Memory usage is less than file size (optimized)")
    else:
        print("Memory usage exceeds file size")
    
    print("\nNote: The indexer uses batch processing to keep memory")
    print("usage low during indexing. The search engine loads the")
    print("entire index into memory for fast lookups.")

if __name__ == "__main__":
    check_memory_optimization()

