# FOR EXTRA CREDIT 1

import hashlib
from collections import defaultdict

class DuplicateDetector:
    def __init__(self):
        self.exact_hashes = {}
        self.content_hashes = {}
        self.simhashes = {}
        self.duplicate_groups = defaultdict(list)
    
    def compute_exact_hash(self, content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def compute_content_hash(self, text: str) -> str:
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def compute_simhash(self, tokens: list[str], hash_bits: int = 64) -> int:
        if not tokens:
            return 0
        
        v = [0] * hash_bits
        for token in tokens:
            h = hash(token)
            for i in range(hash_bits):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint
    
    def hamming_distance(self, hash1: int, hash2: int) -> int:
        return bin(hash1 ^ hash2).count('1')
    
    def is_exact_duplicate(self, content: str, doc_id: int) -> tuple[bool, int]:
        exact_hash = self.compute_exact_hash(content)
        if exact_hash in self.exact_hashes:
            original_doc_id = self.exact_hashes[exact_hash]
            self.duplicate_groups[original_doc_id].append(doc_id)
            return True, original_doc_id
        else:
            self.exact_hashes[exact_hash] = doc_id
            return False, doc_id
    
    def is_near_duplicate(self, text: str, tokens: list[str], doc_id: int, threshold: int = 3) -> tuple[bool, int]:
        content_hash = self.compute_content_hash(text)
        if content_hash in self.content_hashes:
            original_doc_id = self.content_hashes[content_hash]
            self.duplicate_groups[original_doc_id].append(doc_id)
            return True, original_doc_id
        
        simhash = self.compute_simhash(tokens)
        
        for existing_doc_id, existing_simhash in self.simhashes.items():
            distance = self.hamming_distance(simhash, existing_simhash)
            if distance <= threshold:
                self.duplicate_groups[existing_doc_id].append(doc_id)
                return True, existing_doc_id
        
        self.content_hashes[content_hash] = doc_id
        self.simhashes[doc_id] = simhash
        return False, doc_id
    
    def get_duplicate_stats(self) -> dict:
        total_duplicates = sum(len(group) for group in self.duplicate_groups.values())
        unique_originals = len(self.duplicate_groups)
        
        return {
            "unique_original_pages": len(self.exact_hashes),
            "duplicate_groups": unique_originals,
            "total_duplicate_pages": total_duplicates,
            "duplicate_groups_detail": dict(self.duplicate_groups)
        }

