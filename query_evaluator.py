import json
import time
from search import SearchEngine
from test_queries import test_queries

class QueryEvaluator:
    def __init__(self, index_path: str, docids_path: str):
        self.engine = SearchEngine(index_path, docids_path)
        self.results = []
    
    def evaluate_query(self, query: str) -> dict:
        start_time = time.time()
        results = self.engine.search(query, top_k=10)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000
        
        return {
            "query": query,
            "num_results": len(results),
            "response_time_ms": response_time,
            "results": results[:5]
        }
    
    def evaluate_all(self):
        print("Evaluating queries...\n")
        for i, query in enumerate(test_queries, 1):
            result = self.evaluate_query(query)
            self.results.append(result)
            print(f"{i}. Query: '{query}'")
            print(f"   Results: {result['num_results']}, Time: {result['response_time_ms']:.2f}ms")
            if result['results']:
                print(f"   Top result: {result['results'][0][0]}")
            print()
    
    def analyze_performance(self):
        good_queries = []
        poor_queries = []
        
        for result in self.results:
            if result['num_results'] > 0 and result['response_time_ms'] < 100:
                good_queries.append(result)
            else:
                poor_queries.append(result)
        
        print("=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(f"\nGood performing queries ({len(good_queries)}):")
        for r in good_queries[:5]:
            print(f"  - '{r['query']}' ({r['num_results']} results, {r['response_time_ms']:.2f}ms)")
        
        print(f"\nPoor performing queries ({len(poor_queries)}):")
        for r in poor_queries[:10]:
            reason = []
            if r['num_results'] == 0:
                reason.append("no results")
            if r['response_time_ms'] >= 100:
                reason.append("slow")
            print(f"  - '{r['query']}' ({r['num_results']} results, {r['response_time_ms']:.2f}ms) - {', '.join(reason)}")
        
        return good_queries, poor_queries
    
    def save_results(self, filename: str = "query_results.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    evaluator = QueryEvaluator("final_index.json", "final_docids.json")
    evaluator.evaluate_all()
    good, poor = evaluator.analyze_performance()
    evaluator.save_results()

