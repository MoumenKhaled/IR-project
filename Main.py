import subprocess
import time
from flask import Flask, request, jsonify, render_template
import json
import os
import spacy
import requests
from collections import Counter
from Services.Data_Preprocessing import DataPreprocessing
from Services.Data_Indexing import Indexing
from Services.MatchingAndRanking import MatchingAndResult
from Services.Query_Optimization import Query_Optimization
from Services.Evaluation import EvaluationMetrics
from Services.Clustring import Clustering
from Services.AutoComplete import AutoCompleteService
from werkzeug.utils import secure_filename
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS
import pickle 

from sklearn.cluster import KMeans
import spacy
import logging
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def start_flask_service(script_path, port):
    """Function to start a Flask service in a separate process"""
    full_script_path = os.path.join(BASE_DIR, script_path)
    return subprocess.Popen(["flask", "run", "--port={}".format(port)], env=dict(os.environ, FLASK_APP=full_script_path))
data_preprocessing_path = os.path.join(BASE_DIR, "Services", "Data_Preprocessing.py")
data_indexing_path = os.path.join(BASE_DIR, "Services", "Data_Indexing.py")
data_MatchingAndRanking_path = os.path.join(BASE_DIR, "Services", "MatchingAndRanking.py")
data_autocomplete_path = os.path.join(BASE_DIR, "Services", "AutoComplete.py")
data_clustring_path = os.path.join(BASE_DIR, "Services", "Clustring.py")
data_evaluation_path = os.path.join(BASE_DIR, "Services", "Evaluation.py")



data_preprocessing_service = start_flask_service(data_preprocessing_path, 9001)
print("Data Preprocessing Service started on port 9001")
time.sleep(5)  

data_indexing_service = start_flask_service(data_indexing_path, 9002)
print("Data Indexing Service started on port 9002")
time.sleep(5)  

data_Matching_service = start_flask_service(data_MatchingAndRanking_path, 9003)
print("Data MatchingAndRanking Service started on port 9003")
time.sleep(5)  

data_AutoComplete_service = start_flask_service(data_autocomplete_path, 9004)
print("Data MatchingAndRanking Service started on port 9004")
time.sleep(5)  

data_clustring_service = start_flask_service(data_clustring_path, 9006)
print("Clustring Service started on port 9006")
time.sleep(5)  

data_evaluation_service = start_flask_service(data_evaluation_path, 9007)
print("Data Evaluation started on port 9007")
time.sleep(5)  

app = Flask(__name__)
CORS(app)
data_preprocessing = DataPreprocessing()
indexing = Indexing()


logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    dataset_name = request.form['dataset']
    query = request.form['query']
    clean_query = requests.post('http://localhost:9001/clean_query', json={"query": query,"datasetname":dataset_name}).json()['cleaned_query']
    
    tfidf_file = os.path.join(BASE_DIR, dataset_name, 'vectorized_data.json')
    response = requests.post('http://localhost:9002/vectorize_query', json={"query": clean_query, "file_path": tfidf_file})
    query_vector = None
    if response.status_code == 200:
        try:
            query_vector = response.json()['query_vector']
        except ValueError as e:
            print("Error decoding JSON:", e)
            return jsonify({"error": "Error decoding JSON"}), 500
    else:
        print("Error with the request:", response.status_code, response.text)
        return jsonify({"error": "Error with vectorization request", "status_code": response.status_code}), response.status_code

    if query_vector is not None:
        top_10_indices_response = requests.post('http://localhost:9003/match', json={"query_vector": query_vector, "file_path": tfidf_file})
        if top_10_indices_response.status_code == 200:
            top_10_indices = top_10_indices_response.json()['top_10_indices']
        else:
            print("Error with match request:", top_10_indices_response.status_code, top_10_indices_response.text)
            return jsonify({"error": "Error with match request", "status_code": top_10_indices_response.status_code}), top_10_indices_response.status_code
    else:
        return jsonify({"error": "Query vector is not available due to previous errors"}), 500

    document_file_path = os.path.join(BASE_DIR, dataset_name, 'dev', 'collection.tsv')
    results = []
    with open(document_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            id, content = line.split('\t', 1)
            if int(id) in top_10_indices:
                results.append({"id": id, "snippet": content[:200]})  
    autocomplete_dict_path = os.path.join(BASE_DIR,'Services', 'databases', dataset_name, 'autocomplete_dict.json')
    if not os.path.exists(autocomplete_dict_path):
        autocomplete_dict = defaultdict(int)
    else:
        with open(autocomplete_dict_path, 'r') as f:
            try:
                autocomplete_dict = json.load(f)
            except json.JSONDecodeError:
                autocomplete_dict = defaultdict(int)
    terms = query.split()
    for term in terms:
        if term in autocomplete_dict:
            autocomplete_dict[term] += 1
        else:
            autocomplete_dict[term] = 1

    with open(autocomplete_dict_path, 'w') as f:
        json.dump(autocomplete_dict, f)
    return jsonify({"query": query, "results": results})

@app.route('/document/<int:doc_id>')
def document_detail(doc_id):
    dataset_name = request.args.get('dataset')  # You need to decide how to get the correct dataset
    document_file_path = os.path.join(BASE_DIR, dataset_name, 'dev','collection.tsv')
    document_content = ""
    with open(document_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            id, content = line.split('\t', 1)
            if int(id) == doc_id:
                document_content = content
                break
    return render_template('document_detail.html', content=document_content)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    
        dataset_option = request.form['dataset']
        output_file = f"{dataset_option}/vectorized_data.json"

        query_Optimization= Query_Optimization()

        query_Optimization.clean_relevant_query(dataset_option)
        #query_Optimization.tfidf_queries(dataset_option,output_file)

        evaluate_service_url = "http://localhost:9007/evaluate_service"


        data_to_send = {
          "true_data": open(f"{dataset_option}/cleaned_queries.jsonl", 'r').read(),
          "predictions": open(f"{dataset_option}/tfidf_query_results.json", 'r').read()
                      }
        response = requests.post(evaluate_service_url, json=data_to_send)
        if response.status_code == 200:
           metrics = response.json()
           print("metrics",metrics)
           return jsonify(metrics), 200
        else:
            return jsonify({"error": "Failed to retrieve metrics from evaluation service"}), 500
        return jsonify(metrics), 200

@app.route('/offline', methods=['POST'])
def offline():
    dataset_option = request.form['dataset']
    if dataset_option == 'clinical':
        input_file = "clinical/dev/collection.tsv"
        output_file = "clinical/vectorized_data.json"
    elif dataset_option == 'lifestyle':
        input_file = "lifestyle/dev/collection.tsv"
        output_file = "lifestyle/vectorized_data.json"
    
    if os.path.exists(output_file):
        return jsonify({"error": "Offline processing for this dataset has already been done."}), 400

    response = requests.post('http://localhost:9001/process_dataset', json={"dataset_path": input_file})
    if response.status_code == 200:
        try:
             dataset_terms = response.json()['processed_dataset']
        except ValueError:
               return jsonify({"error": "Invalid JSON response received"}), 500
    else:
         return jsonify({"error": "Failed to process dataset"}), response.status_code
    
    requests.post('http://localhost:9002/vectorize_data', json={"dataset_terms": dataset_terms,"output_file": output_file}).json()['message']
    return jsonify({"message": "Offline processing complete. TF-IDF data stored."})
if __name__ == '__main__':
    app.run(debug=True, port=9000) 

