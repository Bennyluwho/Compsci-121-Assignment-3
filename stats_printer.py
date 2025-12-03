import json
import os

class StatsPrinter:
    def print_index_stats(self, index_path, docid_path):
        #highkey finished :)
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        with open(docid_path, "r", encoding="utf-8") as f:
            docids = json.load(f)
        
        unique_tokens = len(index)
        num_documents = len(docids)
        
        print(f"unique tokens: {unique_tokens}")
        print(f"indexed pages: {num_documents}")