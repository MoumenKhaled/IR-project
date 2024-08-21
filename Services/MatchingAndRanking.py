from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

class MatchingAndResult:
    def matching(self, tfidf_matrix, query_vector):
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
        relevant_doc_indices = similarity_scores.argsort()[0][::-1]
        top_10_indices = relevant_doc_indices[:10]
        return top_10_indices

matching_service = MatchingAndResult()

@app.route('/match', methods=['POST'])
def match():
    query_vector = np.array(request.json['query_vector'])
    file_path = request.json['file_path']
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    tfidf_matrix = data['tfidf_matrix']
    top_10_indices = matching_service.matching(tfidf_matrix, query_vector)
    return jsonify({"top_10_indices": top_10_indices.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=9003)
