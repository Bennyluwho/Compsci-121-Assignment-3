#TODO: Number of unique tokens
#TODO: Total size (in KB) of your index on disk
#TODO: OPTIONAL save the time it takes

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings
import json
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from collections import defaultdict
from postings import Posting
from duplicate_detector import DuplicateDetector
from urllib.parse import urljoin, urlparse

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)



class Indexer:
    def __init__(self, root_folder: str, batch_size: int = 3000, detect_duplicates: bool = True):
        self.root = Path(root_folder)
        self.batch_size = batch_size
        self.detect_duplicates = detect_duplicates

        self.tokenizer = RegexpTokenizer(r"[a-zA-Z0-9]+")
        # Porter Stemming
        self.stemmer = PorterStemmer()

        self.inverted_index = defaultdict(list) # stores a list of posting  objects
        self.doc_id_to_url = {}
        # Global anchor words map: {target_url: [anchor_tokens]}
        self.global_anchor_words = defaultdict(list)
        # Global URL to doc_id mapping (across all batches)
        self.global_url_to_doc_id = {}

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
        
        # important words
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
    
    # TODO: EXTRA CREDIT 5
    def extract_anchor_words(self, html: str, base_url: str) -> dict[str, list[str]]:
        anchor_map = defaultdict(list)
        
        if not html or not html.strip():
            return anchor_map
        
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return anchor_map
        
        try:
            for link in soup.find_all("a", href=True):
                href = link.get("href", "").strip()
                anchor_text = link.get_text(strip=True)
                
                if not href or not anchor_text:
                    continue
                
                # Resolve relative URLs to absolute URLs
                try:
                    absolute_url = urljoin(base_url, href)
                    # Normalize URL (remove fragment, normalize)
                    parsed = urlparse(absolute_url)
                    normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    if parsed.query:
                        normalized_url += f"?{parsed.query}"
                    
                    # Tokenize and stem anchor text
                    anchor_tokens = self.tokenize_and_stem(anchor_text)
                    if anchor_tokens:
                        anchor_map[normalized_url].extend(anchor_tokens)
                except Exception:
                    continue
        except Exception:
            pass
        
        return anchor_map
    
    #index single document

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
    
    def index_anchor_words(self, target_doc_id: int, anchor_tokens: list[str]) -> None:
        for token in anchor_tokens:
            posting_list = self.inverted_index[token]
            
            found = False
            for posting in posting_list:
                if posting.doc_id == target_doc_id:
                    # increase posting for anchor words
                    posting.increment(is_important=True)
                    found = True
                    break
            if not found:
                # create postings for anchor words
                posting_list.append(Posting(target_doc_id, 1, 1))


    def batch_grab(self):
        batch = []
        for file_path in self.root.rglob("*.json"):
            batch.append(file_path)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        # trying to catch remaining files
        if batch:
            yield batch

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
                self.global_url_to_doc_id[url] = doc_id
                self.index_document(doc_id, html)
                
                # Extract anchor words from this document and store globally
                anchor_map = self.extract_anchor_words(html, url)
                for target_url, anchor_tokens in anchor_map.items():
                    self.global_anchor_words[target_url].extend(anchor_tokens)
                    
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


    def process_anchor_words(self):
        """
        Process all collected anchor words and index them to target documents.
        This needs to update all partial indexes.
        """
        print("Processing anchor words...")
        anchor_count = 0
        
        # Process anchor words for each target URL that exists in our index
        for target_url, anchor_tokens in self.global_anchor_words.items():
            if target_url in self.global_url_to_doc_id:
                target_doc_id = self.global_url_to_doc_id[target_url]
                # Find which batch this doc_id belongs to by checking partial indexes
                # We need to update the partial index that contains this doc_id
                # For simplicity, we'll update all partial indexes that might contain the target
                anchor_count += len(anchor_tokens)
        
        # Load all partial indexes and update them with anchor words
        partial_index_files = sorted(Path(".").glob("partial_index_*.json"))
        
        for index_file in partial_index_files:
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    partial_index = json.load(f)
                
                # Load corresponding docids to find which URLs are in this partial index
                batch_id = int(index_file.stem.split("_")[-1])
                docid_file = Path(f"partial_docids_{batch_id}.json")
                
                if docid_file.exists():
                    with open(docid_file, "r", encoding="utf-8") as f:
                        partial_docids = json.load(f)
                    
                    # Create reverse mapping for this batch
                    url_to_doc_id = {v: int(k) for k, v in partial_docids.items()}
                    
                    # Update index with anchor words for URLs in this batch
                    for target_url, anchor_tokens in self.global_anchor_words.items():
                        if target_url in url_to_doc_id:
                            target_doc_id = url_to_doc_id[target_url]
                            # Index anchor words to this document
                            for token in anchor_tokens:
                                if token not in partial_index:
                                    partial_index[token] = []
                                
                                # Find or create posting for this doc_id
                                found = False
                                for posting in partial_index[token]:
                                    if posting["doc_id"] == target_doc_id:
                                        posting["tf"] += 1
                                        posting["imp"] += 1  # Anchor words are important
                                        found = True
                                        break
                                
                                if not found:
                                    partial_index[token].append({
                                        "doc_id": target_doc_id,
                                        "tf": 1,
                                        "imp": 1
                                    })
                    
                    # Save updated partial index
                    with open(index_file, "w", encoding="utf-8") as f:
                        json.dump(partial_index, f)
            except Exception as e:
                print(f"Error processing anchor words for {index_file}: {e}")
                continue
        
        print(f"Indexed {anchor_count} anchor word tokens across all documents.")

    #RUN
    def build(self):
        batch_id = 0
        for batch_files in self.batch_grab():
            print(f"Processing batch {batch_id} with {len(batch_files)} files.")
            self.process_batch(batch_files, batch_id)
            batch_id += 1
        
        # Process anchor words after all batches are processed
        if self.global_anchor_words:
            self.process_anchor_words()
        
        if self.detect_duplicates and self.duplicate_detector:
            stats = self.duplicate_detector.get_duplicate_stats()
            print(f"Unique original pages: {stats['unique_original_pages']}")
            print(f"Duplicate groups found: {stats['duplicate_groups']}")
            print(f"Total duplicate pages skipped: {stats['total_duplicate_pages']}")
            print(f"Total pages skipped: {self.skipped_duplicates}")

if __name__ == "__main__":
    indexer = Indexer(root_folder="analyst", batch_size=3000)
    indexer.build()

