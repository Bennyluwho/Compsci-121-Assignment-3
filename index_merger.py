class IndexMerger:
    def merge_partial_indexes(self, partial_index_paths: list[str]):
        #TODO: read each partial index
            #: for each token, combine each list of postings object
            #: sum TFs if the same doc_id appears in multiple batches
        #NOTE: merge before sorting, it was on the quiz :p
        pass

    def save_final_index(self, index_path):
        #saves final merged index to a json file
        pass
    