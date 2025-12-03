
#TODO: Number of unique tokens
#TODO: Total size (in KB) of your index on disk
#TODO: OPTIONAL save the time it takes

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings
import json
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from collections import defaultdict
from postings import Posting
from duplicate_detector import DuplicateDetector

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)



class Indexer:
    def __init__(self, root_folder: str, batch_size: int = 3000, detect_duplicates: bool = True):
        self.root = Path(root_folder)
        self.batch_size = batch_size
        self.detect_duplicates = detect_duplicates

        self.tokenizer = RegexpTokenizer(r"[a-zA-Z0-9]+")
        self.stemmer = SnowballStemmer("english") # swtiched to a faster stemmer

        self.inverted_index = defaultdict(list) # stores a list of posting  objects
        self.doc_id_to_url = {}

        self.global_doc_id = 0
        self.duplicate_detector = DuplicateDetector() if detect_duplicates else None
        self.skipped_duplicates = 0

     # TEXT PROCESSING   

    def extract_text(self, html:str) -> tuple[str, set[str]]:
        if not html or not html.strip():
            return "", set()
        
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return "", set()
        
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        
        important_words = set()
        
        try:
            for tag in soup.find_all(["h1", "h2", "h3", "title"]):
                text = tag.get_text()
                tokens = self.tokenizer.tokenize(text.lower())
                stems = [self.stemmer.stem(t) for t in tokens]
                important_words.update(stems)
            
            for tag in soup.find_all(["strong", "b"]):
                text = tag.get_text()
                tokens = self.tokenizer.tokenize(text.lower())
                stems = [self.stemmer.stem(t) for t in tokens]
                important_words.update(stems)
        except Exception:
            pass
        
        try:
            text = soup.get_text(separator=" ", strip=True)
            return text, important_words
        except Exception:
            return "", important_words

    def tokenize_and_stem(self, text: str) -> list[str]:
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)
        tokens = [t for t in tokens] 
        stems = [self.stemmer.stem(t) for t in tokens]
        return stems
    
    #INDEXING SINGLE DOC

    def index_document(self, doc_id: int, html: str) -> None:
        text, important_words = self.extract_text(html)
        stemmed_tokens = self.tokenize_and_stem(text)

        for token in stemmed_tokens:
            posting_list = self.inverted_index[token]
            is_important = token in important_words
            
            found = False
            for posting in posting_list:
                if posting.doc_id == doc_id:
                    posting.increment(is_important)
                    found = True
                    break
            if not found:
                posting_list.append(Posting(doc_id, 1, 1 if is_important else 0))


    #GRAB BATCHES
    def batch_grab(self):
        batch = []
        for file_path in self.root.rglob("*.json"):
            batch.append(file_path)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        #note: trying to catch leftovers
        if batch:
            yield batch

    #BATCH PROCESSING
    def process_batch(self, batch_files, batch_id):
        self.inverted_index = defaultdict(list)
        self.doc_id_to_url = {}

        for json_file in batch_files:
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                url = data.get("url")
                html = data.get("content", "")

                if not url or not html:
                    continue

                doc_id = self.global_doc_id
                
                if self.detect_duplicates and self.duplicate_detector:
                    is_duplicate, original_doc_id = self.duplicate_detector.is_exact_duplicate(html, doc_id)
                    if is_duplicate:
                        self.skipped_duplicates += 1
                        continue
                    
                    text, _ = self.extract_text(html)
                    stemmed_tokens = self.tokenize_and_stem(text)
                    is_near_dup, original_doc_id = self.duplicate_detector.is_near_duplicate(text, stemmed_tokens, doc_id)
                    if is_near_dup:
                        self.skipped_duplicates += 1
                        continue

                self.global_doc_id += 1
                self.doc_id_to_url[doc_id] = url
                self.index_document(doc_id, html)
            # handle broken or missing HTML
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                continue
            except Exception:
                continue

        #save after finishing with batch
        self.save_partial(batch_id)

    #PARTIAL INDEX
    def save_partial(self, batch_id):
        index_path = f"partial_index_{batch_id}.json"
        docid_path = f"partial_docids_{batch_id}.json"

        serializable_index = { token: [posting.to_dict() for posting in postings]
                              for token, postings in self.inverted_index.items() }
        with open(index_path, "w") as f:
            json.dump(serializable_index, f)
        with open(docid_path, "w") as f:
            json.dump(self.doc_id_to_url, f)


    #RUN
    def build(self):
        batch_id = 0
        for batch_files in self.batch_grab():
            print(f"Processing batch {batch_id} with {len(batch_files)} files.")
            self.process_batch(batch_files, batch_id)
            batch_id += 1
        
        if self.detect_duplicates and self.duplicate_detector:
            stats = self.duplicate_detector.get_duplicate_stats()
            print(f"Unique original pages: {stats['unique_original_pages']}")
            print(f"Duplicate groups found: {stats['duplicate_groups']}")
            print(f"Total duplicate pages skipped: {stats['total_duplicate_pages']}")
            print(f"Total pages skipped: {self.skipped_duplicates}")

if __name__ == "__main__":
    indexer = Indexer(root_folder="analyst", batch_size=3000)
    indexer.build()

