import json
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

class Clustering:
    def __init__(self, num_clusters=3):
        self.vectorizer = TfidfVectorizer()
        self.num_clusters = num_clusters
    def train_kmeans(self, tfidf_matrix, datasetName):
        model_file = f'{datasetName}/kmeans_model.pkl'
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(tfidf_matrix)

        # Save the trained KMeans model to a pickle file
        data = {
            "kmeans_model": kmeans,
            "tfidf_matrix": tfidf_matrix,
        }
        with open(model_file, "wb") as f:
            pickle.dump(data, f)

    def predict_and_get_similar_documents(self, query_tfidf, datasetName):
        model_file = f'{datasetName}/kmeans_model.pkl'
        # Load the trained KMeans model from the pickle file
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
            kmeans_model = data['kmeans_model']
            tfidf_matrix = data['tfidf_matrix']

        query_cluster = kmeans_model.predict(query_tfidf)
        print(query_cluster)

        # Calculating Cosine Similarity
        cluster_mask = kmeans_model.labels_ == query_cluster[0]
        cluster_documents = tfidf_matrix[cluster_mask]
        similarities = cosine_similarity(query_tfidf, cluster_documents)

        # Display the most similar documents
        most_similar_docs_indices = np.argsort(similarities[0])[::-1][:10]
        similar_documents = []
        for index in most_similar_docs_indices:
            similar_documents.append((index, similarities[0][index]))
            print(f"Document {index} Similarity: {similarities[0][index]}")

        return similar_documents

    def visualize_clusters(self, datasetName):
        model_file = f'{datasetName}/kmeans_model.pkl'
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
            kmeans_model = data['kmeans_model']
            tfidf_matrix = data['tfidf_matrix']

        try:
            if kmeans_model is None:
                raise ValueError("KMeans model has not been fitted yet. Call 'fit_kmeans' first.")

            logging.debug("Starting visualization.")
            svd = TruncatedSVD(n_components=2)
            reduced_data = svd.fit_transform(tfidf_matrix)

            plt.figure()  # Start a new figure
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_model.labels_)
            plt.title('Cluster visualization')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.colorbar()  # Show color scale
            plt.show()

            # Ensure the directory exists
            output_dir = 'static'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the figure to a file
            plt.savefig(os.path.join(output_dir, 'cluster_plot.png'))
            plt.close()  # Close the figure to free up memory

            logging.debug("Visualization saved to {}/cluster_plot.png".format(output_dir))
        except Exception as e:
            logging.error("Error in visualization process: %s", e)
            raise

# Instantiate the Clustering class
clustering = Clustering(num_clusters=3)

@app.route('/train_kmeans', methods=['POST'])
def train_kmeans_route():
    data = request.json
    dataset_name = data['dataset']
    base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(base_path, "..") 
    tf_idf_file = os.path.join(base_path, dataset_name, "vectorized_data.json")
   
    with open(tf_idf_file, "rb") as f:
        data = pickle.load(f)
    tfidf_matrix = np.array(data['tfidf_matrix'])
    model_file = f'{dataset_name}/kmeans_model.pkl'
    if os.path.exists(model_file):
        return jsonify({"message": "Model already trained"}), 200
            
    clustering.train_kmeans(tfidf_matrix, dataset_name)
    return jsonify({"message": "Model trained successfully"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    query_vector = np.array(data['query_vector']).reshape(1, -1)
    dataset_name = data['dataset']
    similar_documents = clustering.predict_and_get_similar_documents(query_vector, dataset_name)
    return jsonify({"similar_documents": similar_documents}), 200

@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json
    dataset_name = data['dataset']
    clustering.visualize_clusters(dataset_name)
    return jsonify({"message": "Cluster visualization created successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=9006)
