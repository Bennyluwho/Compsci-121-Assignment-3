import json
import math
from collections import defaultdict
from pathlib import Path
from scoring import compute_idf, compute_tf_idf

def compute_magnitudes(index_path: list[str], total_docs: int, output_path: str):
 
    doc_magnitudes = defaultdict(float)
    for index_file in index_path:
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        for token, postings in index.items():
            df = len(postings)
            idf = compute_idf(total_docs, df)

            if idf == 0.0:
                continue

            #compute tf-idf for each posting
            for postings in postings:
                doc_id = postings["doc_id"]
                tf = postings["tf"]
                imp = postings.get("imp", 0)

                doc_magnitudes[doc_id] += compute_tf_idf(tf, idf, imp) ** 2

    doc_magnitudes = {doc_id: math.sqrt(magnitude) for doc_id, magnitude in doc_magnitudes.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_magnitudes, f)
    
    return output_path