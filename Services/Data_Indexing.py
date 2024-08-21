from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)

class Indexing:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def data_vectorize(self, dataset_terms):
        tfidf_matrix = self.vectorizer.fit_transform(dataset_terms)
        return tfidf_matrix

    def query_vectorize(self, query):
        query_vector = self.vectorizer.transform([query])
        return query_vector
    def query_vectorize_C(self, query, file_path):
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading file: {e}")
            raise
        vectorizer = data["vectorizer"]
        tfidf_matrix = data["tfidf_matrix"]

        # Vectorize the query
        query_vector = vectorizer.transform([query])
        return tfidf_matrix,query_vector
indexing_service = Indexing()

@app.route('/vectorize_data', methods=['POST'])
def vectorize_data():
    data = request.json
    dataset_terms = data['dataset_terms']
    tfidf_matrix = indexing_service.data_vectorize(dataset_terms)
    file_path = data['output_file']
    with open(file_path, "wb") as f:
        pickle.dump({'vectorizer': indexing_service.vectorizer, 'tfidf_matrix': tfidf_matrix}, f)
    return jsonify({"message": "Data vectorized successfully"})


@app.route('/vectorize_query', methods=['POST'])
def vectorize_query():
    try:
        query = request.json['query']
        file_path = request.json['file_path']
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        indexing_service.vectorizer = data['vectorizer']
        query_vector = indexing_service.query_vectorize(query)
        expected_features = data['vectorizer'].vocabulary_.__len__()  
        if query_vector.shape[1] != expected_features:
            return jsonify({"error": "Dimension mismatch in query vectorization"}), 500
        
        return jsonify({"query_vector": query_vector.toarray().tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, port=9002)
