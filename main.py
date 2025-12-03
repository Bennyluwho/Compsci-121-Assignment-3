from indexer import Indexer
from index_merger import IndexMerger
from stats_printer import StatsPrinter
from pathlib import Path
import time
import json

def clean_old_partials(folder="."):
    folder = Path(folder)
    for f in folder.glob("partial_index_*.json"):
        f.unlink()
    for f in folder.glob("partial_docids_*.json"):
        f.unlink()
    print("Old partial JSON files deleted.")



if __name__ == "__main__":

    clean_old_partials(".")

    start_index_time = time.time()
    
    indexer = Indexer(root_folder="DEV", batch_size=2000)
    indexer.build()

    end_index_time = time.time()
    indexing_duration = end_index_time - start_index_time
    print(f"Indexing completed in {indexing_duration:.2f} seconds.")
    print(f"Total pages indexed: {indexer.global_doc_id}")

    print(f"Number of indexed pages:", indexer.global_doc_id)
    print(f"Number of unique tokens:", len(indexer.inverted_index))

    partial_index_files = sorted(Path(".").glob("partial_index_*.json"))
    partial_docid_files = sorted(Path(".").glob("partial_docids_*.json"))
    
    if partial_index_files:
        start_merge_time = time.time()
        
        merger = IndexMerger()
        merger.merge_partial_indexes([str(f) for f in partial_index_files])
        merger.merge_partial_docids([str(f) for f in partial_docid_files])
        merger.save_final_index("final_index.json")
        
        with open("final_docids.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in merger.merged_docids.items()}, f)
        
        end_merge_time = time.time()
        merge_duration = end_merge_time - start_merge_time
        
        print("\nMerged all partial indexes.")
        print(f"Merging completed in {merge_duration:.2f} seconds.")
        
        print("\nSplitting index by term ranges...")
        manifest_path = merger.split_index_by_term_ranges("final_index.json", num_splits=10)
        print(f"Index split complete. Manifest: {manifest_path}")
        
        stats = StatsPrinter()
        stats.print_index_stats("final_index.json", "final_docids.json")
    else:
        print("Nothing to merge")