from indexer import Indexer
from index_merger import IndexMerger
from stats_printer import StatsPrinter

if __name__ == "__main__":
    indexer = Indexer(root_folder="analyst", batch_size=3000)
    indexer.build()