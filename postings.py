
class Posting:
    def __init__(self, doc_id: int, term_freq: int = 1, important_freq: int = 0):
        self.doc_id = doc_id
        self.term_freq = term_freq
        self.important_freq = important_freq

    def increment(self, is_important: bool = False):
        self.term_freq += 1
        if is_important:
            self.important_freq += 1

    def to_dict(self):
        return {"doc_id": self.doc_id, "tf": self.term_freq, "imp": self.important_freq}