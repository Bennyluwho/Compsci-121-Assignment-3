from indexer import Indexer
from index_merger import IndexMerger
from stats_printer import StatsPrinter
from pathlib import Path
import time

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

    indexer = Indexer(root_folder="DEV", batch_size=20000)
    indexer.build()

    end_index_time = time.time()
    indexing_duration = end_index_time - start_index_time

    print(f"Indexing completed in {indexing_duration:.2f} seconds.")
    print(f"Total pages indexed: {indexer.global_doc_id}")
  
    #HACK: QUICK FIX, WONT WONT WITH UNMERGED PARTIAL INDEXES
    start_merge_time = time.time()
    
    partial_index_paths = sorted(
        str(p) for p in Path(".").glob("partial_index_*.json")
    )

    partial_docid_paths = sorted(
        str(p) for p in Path(".").glob("partial_docids_*.json")
    )

    merger = IndexMerger()
    merger.merge_partial_indexes(partial_index_paths)
    merger.merge_partial_docids(partial_docid_paths)
    merger.save_final_index("final_index.json")
    merger.save_final_docids("final_docids.json")

    end_merge_time = time.time()
    merge_duration = end_merge_time - start_merge_time

    print(f"Merging completed in {merge_duration:.2f} seconds.")