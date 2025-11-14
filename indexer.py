
#TODO: Number of unique tokens
#TODO: Total size (in KB) of your index on disk
#TODO: OPTIONAL save the time it takes

from bs4 import BeautifulSoup
import json
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from collections import defaultdict
import time
from postings import Posting



class Indexer:
    def __init__(self, root_folder: str, batch_size: int = 3000):
        self.root = Path(root_folder)
        self.batch_size = batch_size

        self.tokenizer = RegexpTokenizer(r"\w+")
        self.stemmer = SnowballStemmer("english") # swtiched to a faster stemmer

        self.inverted_index = defaultdict(list) # stores a list of posting  objects
        self.doc_id_to_url = {}

        self.global_doc_id = 0

     # TEXT PROCESSING   

    def extract_text(self, html:str) -> str:
        if not html.strip():
            return ""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)

    def tokenize_and_stem(self, text: str) -> list[str]:
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)
        tokens = [t for t in tokens if t.isalpha() if t.isalpha()] 
        stems = [self.stemmer.stem(t) for t in tokens]
        return stems
    
    #INDEXING SINGLE DOC

    def index_document(self, doc_id: int, html: str) -> None:
        text = self.extract_text(html)
        stemmed_tokens = self.tokenize_and_stem(text)

        for token in stemmed_tokens:
            posting_list = self.inverted_index[token]
            #check if doc_id has a posting

            found = False
            for posting in posting_list:
                if posting.doc_id == doc_id:
                    posting.increment()
                    found = True
                    break
            if not found:
                posting_list.append(Posting(doc_id))


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
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

                url = data.get("url")
                html = data.get("content", "")

                if not url or not html:
                    continue

                doc_id = self.global_doc_id
                self.global_doc_id += 1

                self.doc_id_to_url[doc_id] = url
                self.index_document(doc_id, html)

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

        print(f"Saved batch {batch_id} to {index_path}")

    #RUN
    def build(self):
        batch_id = 0
        for batch_files in self.batch_grab():
            print(f"Processing batch {batch_id} with {len(batch_files)} files.")
            self.process_batch(batch_files, batch_id)
            batch_id += 1

if __name__ == "__main__":
    indexer = Indexer(root_folder="DEV", batch_size=3000)
    indexer.build()

