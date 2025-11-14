
class Posting:
    def __init__(self, doc_id: int, term_freq: int = 1):
        self.doc_id = doc_id
        self.term_freq = term_freq

    def increment(self):
        self.term_freq += 1

    def to_dict(self):
        return {"doc_id": self.doc_id, "tf": self.term_freq}