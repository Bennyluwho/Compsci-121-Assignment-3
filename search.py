import json
from pathlib import Path
from typing import List, Dict

from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer


class SearchEngine:
    def __init__(
        self,
        index_path: str = "final_index.json",
        docids_path: str = "final_docids.json",
    ):
        """
        Loads the merged inverted index and docid->URL mapping.
        """
        self.tokenizer = RegexpTokenizer(r"[a-zA-Z0-9]+")   # match indexer
        self.stemmer = SnowballStemmer("english")

        index_file = Path(index_path)
        docids_file = Path(docids_path)

        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not docids_file.exists():
            raise FileNotFoundError(f"DocID mapping file not found: {docids_file}")

        with index_file.open("r", encoding="utf-8") as f:
            # index: token -> list of {"doc_id": int, "tf": int}
            self.index: Dict[str, List[Dict[str, int]]] = json.load(f)

        with docids_file.open("r", encoding="utf-8") as f:
            # docid_to_url: str(doc_id) -> url
            self.docid_to_url: Dict[str, str] = json.load(f)

        # total number of documents, useful later for tf-idf
        self.N = len(self.docid_to_url)

    # Text processing
    def tokenize_and_stem(self, text: str) -> List[str]:
        """
        Tokenize and stem the query using the SAME pipeline as the indexer.
        """
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)
        stems = [self.stemmer.stem(t) for t in tokens]
        return stems

    # Core retrieval (AND queries)
    def _get_posting_docids(self, token: str) -> List[int]:
        """
        Returns the list of doc_ids for a given token.
        If the token is not in the index, returns an empty list.
        """
        postings = self.index.get(token, [])
        return [p["doc_id"] for p in postings]

    def and_query(self, query: str) -> List[int]:
        """
        Processes an AND-only query:
        - tokenize + stem the query
        - retrieve postings for each term
        - intersect doc_id sets
        Returns a sorted list of matching doc_ids.
        """
        terms = self.tokenize_and_stem(query)
        if not terms:
            return []

        # Get posting doc_id lists for each term
        posting_lists: List[List[int]] = [
            self._get_posting_docids(term) for term in terms
        ]

        # If any term has no documents -> empty result
        if any(len(pl) == 0 for pl in posting_lists):
            return []

        # Intersect using sets (simple and fine for this assignment)
        posting_lists.sort(key=len)  # start with smallest for efficiency
        result_set = set(posting_lists[0])
        for pl in posting_lists[1:]:
            result_set &= set(pl)

        return sorted(result_set)

    # Helper: get top K URLs
    def top_k_urls(self, doc_ids: List[int], k: int = 5) -> List[str]:
        """
        Given a list of doc_ids, return up to k corresponding URLs.
        """
        urls: List[str] = []
        for doc_id in doc_ids[:k]:
            url = self.docid_to_url.get(str(doc_id), f"[missing url for doc {doc_id}]")
            urls.append(url)
        return urls


if __name__ == "__main__":
    # Simple CLI interface
    engine = SearchEngine(index_path="final_index.json", docids_path="final_docids.json")

    print("Simple AND-only search. Empty query to quit.")
    while True:
        q = input("Query> ").strip()
        if not q:
            break

        matching_docs = engine.and_query(q)
        urls = engine.top_k_urls(matching_docs, k=5)

        print(f"\nQuery: {q}")
        print(f"Matched {len(matching_docs)} documents. Top {min(5, len(matching_docs))}:")
        for i, url in enumerate(urls, start=1):
            print(f"{i}. {url}")
        print()