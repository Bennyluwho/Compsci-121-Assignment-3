from flask import Flask, render_template, request, jsonify
from search import SearchEngine

app = Flask(__name__)

# initialize search engine
engine = SearchEngine("final_index.json", "final_docids.json", "doc_magnitudes.json")

@app.route('/')
def index():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    top_k = request.json.get('top_k', 10)
    
    if not query:
        return jsonify({'results': [], 'message': 'Please enter a search query'})
    
    results = engine.search(query, top_k=top_k)
    
    formatted_results = []
    for url, score in results:
        formatted_results.append({
            'url': url,
            'score': round(score, 4)
        })
    
    return jsonify({
        'results': formatted_results,
        'count': len(formatted_results)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

