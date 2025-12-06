import math

def compute_idf(total_docs: int, doc_freq: int) -> float:
    if doc_freq == 0:
        return 0.0
    return math.log(total_docs / doc_freq)

def compute_tf_idf(tf: int, idf: float, imp: int = 0) -> float:
    base_score = tf * idf
    important_boost = imp * idf * 1.5
    return base_score + important_boost
