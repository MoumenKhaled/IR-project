إعداد الطلاب : مؤمن خالد ، أحمد علاء الدين ، وفاء جبرائيل ، ناصر دحدل ، رهف الملحم

.توصيف الطلبات الأساسية والخطوات الرئيسية لكل service :

1.	Process Dataset & Clean Query Services:

وظيفة هذه الـ services هي تنظيف الداتا والكويري تجهيزا لبناء الـindex  علما ان الدخل الخاص بالـ service Process Dataset هو مسار ملف الـ  data setوالخرج هو عبارة عن  data  terms والدخل الخاص بالـ Clean Query Service هو الـ query والخرج هو clean query كلا الخرجين مطبق عليهم العمليات التالية: 
•	Remove phonetic notation
•	tokenize text 
•	convert to lowercase 
•	remove urls
•	remove punctuation
•	process and standardize dates brackets
•	remove stopwords
•	stem_tokens
•	lemmatize_words
تم تطبيق العمليات حسب نوع الداتا سيت كما هو موضح بالكود التالي:

•	الكود :

    def process_text_lifestyle(self, text):
        text_without_phonetic = self.remove_phonetic_notation(text)
        tokens = self.tokenize_text(text_without_phonetic)
        tokens = self.convert_to_lowercase(tokens)
        tokens = self.remove_urls(tokens)
        tokens = self.remove_punctuation(tokens)
        tokens = self.process_and_standardize_dates_brackets(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_tokens(tokens)
        tokens = self.lemmatize_words(tokens)
        return self.join_tokens(tokens)

    def process_text_clinical(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens
    
    def data_set_process(self, data_set_path, datasetname):
        max_lines = 50000
        dataset_terms = []

        with open(data_set_path, 'r', encoding='utf-8') as file:
            data = file.read().split("\n")
            for i, row in enumerate(data):
                if i >= max_lines:
                    break
                if row.strip():
                    document = ' '.join(row.split(' ')[1:])
                    if datasetname == 'lifestyle':
                        processed_text = self.process_text_lifestyle(document)
                    else:
                        processed_text = self.process_text_clinical(document)
                        stemmed_text = self.stem_tokens(processed_text)
                        lemmatized_text = self.lemmatize_words(stemmed_text)
                        processed_text = ' '.join(lemmatized_text)
                    dataset_terms.append(processed_text)
        return dataset_terms
def clean_query(self, query, datasetname):
        if datasetname == 'lifestyle':
            processed_text = self.process_text_lifestyle(query)
        else:
            processed_text = self.process_text_clinical(query)
            stemmed_text = self.stem_tokens(processed_text)
            lemmatized_text = self.lemmatize_words(stemmed_text)
            processed_text = ' '.join(lemmatized_text)
        return processed_text

data_preprocessor = DataPreprocessing()

@app.route('/clean_query', methods=['POST'])
def clean_query():
    data = request.json
    cleaned_query = data_preprocessor.clean_query(data['query'], data['datasetname'])
    return jsonify({"cleaned_query": cleaned_query})

@app.route('/process_dataset', methods=['POST'])
def process_dataset():
    data = request.json
    dataset_terms = data_preprocessor.data_set_process(data['dataset_path'], data['datasetname'])
    return jsonify({"processed_dataset": dataset_terms})



















2.	Vectorize Data Service:
وظيفة هذه الـ service هي بناء الـ tf-idf matrix للداتا ، الدخل الخاص بهذه الـ service هو الـ  data terms وتقوم بتخزين الـ  tf-idf-matrixو الـ  vectorizerبملف json باسم vectorized_data.json :
•	الكود : 


    def data_vectorize(self, dataset_terms):
        tfidf_matrix = self.vectorizer.fit_transform(dataset_terms)
        return tfidf_matrix
@app.route('/vectorize_data', methods=['POST'])
def vectorize_data():
    data = request.json
    dataset_terms = data['dataset_terms']
    tfidf_matrix = indexing_service.data_vectorize(dataset_terms)
    file_path = data['output_file']
    with open(file_path, "wb") as f:
        pickle.dump({'vectorizer': indexing_service.vectorizer, 'tfidf_matrix': tfidf_matrix}, f)
    return jsonify({"message": "Data vectorized successfully"})











3.	Vectorize Query Service:

هذه الـ service  تقوم بتحميل الـ vectorizer و tf-idf matrix الذي تم تخزينهم في الـ service  السابقة وحساب الـ tf-idf  للكويري بناء عليها يتم حساب الـ tf-idf matrix الخاص بالـ query ، دخل هذه الـ service  هو الـ query  بعد التنظيف والخرج هو tf-idf matrix الخاص بالـ query باسم query_vector و الـ tf-idf matrix الخاص بالـ Data set تجهزا لعملية حساب الـ cosine_similarity في الـ service  القادمة. 
•	الكود : 

 def query_vectorize(self, query):
        query_vector = self.vectorizer.transform([query])
        return query_vector
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






4.	Matching and Result Service:
في هذه  الـ service  يتم حساب الـ cosine_similarity للـ tf-idf matrix الخاص بالـ query و الـ tf-idf matrix الخاص بالـ Data set وإعادة المستندات المناسبة وترتيبها من الأعلى للأقل ، دخل هذه الـ service هو tf-idf matrix الخاص بالـ query باسم query_vector و الـ tf-idf matrix الخاص بالـ Data set والخرج هو المستندات الأعلى تشابه والغير صفرية.
•	الكود: 

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



5.	Evaluate Service:
 في هذه الـ service يتم حساب معايير التقييم التالية :

•	Mean Average Precision (MAP)
•	Recall
•	Precision@10
•	Mean Reciprocal Rank (MRR)
مدخل هذه الخدمة هو مسار لمفين: 
-	الأول هو الملف الذي يحوي على النتائج التي يجب ان يعيدها النظام.
-	الثاني هو الملف الذي يحوي على النتائج التي يعيدها النظام باسم.
وعن طريق المقارنة يتم حساب معايير التقييم السابقة وطباعتها ، الخرج هو نتائج معايير التقييم السابقة.
•	الكود : 

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

class EvaluationMetrics:
    def __init__(self):
        pass  # Initialization does not need to preload data

    def load_data(self, data):
        return [json.loads(line)['answer_pids'] for line in data.splitlines()]
        
    def load_predictions(self, data):
        return [entry['sorted_non_zero_indices'] for entry in json.loads(data)]

    def calculate_recall(self, true_pids, pred_indices):
        true_set = set(true_pids)
        pred_set = set(pred_indices)
        if len(true_set) == 0:
            return 0
        return len(true_set & pred_set) / len(true_set)

    def calculate_precision_at_k(self, true_pids, pred_indices, k):
        true_set = set(true_pids)
        pred_set = set(pred_indices[:k])
        if len(pred_set) == 0:
            return 0
        return len(true_set & pred_set) / k

    def average_precision(self, true_pids, pred_indices):
        relevant = 0
        sum_precisions = 0
        for i, pred in enumerate(pred_indices):
            if pred in true_pids:
                relevant += 1
                sum_precisions += relevant / (i + 1)
        if relevant == 0:
            return 0
        return sum_precisions / len(true_pids)
    def calculate_reciprocal_rank(self, true_pids, pred_indices):
        for i, pred in enumerate(pred_indices):
          if pred in true_pids:
            return 1 / (i + 1)
        return 0
    def calculate_metrics(self, true_data, predictions):
        recalls = []
        precisions_k = []
        aps = []
        mrrs = []  # لتخزين نتائج MRR لكل مثال

        for true_ids, pred_ids in zip(true_data, predictions):
            recalls.append(self.calculate_recall(true_ids, pred_ids))
            precisions_k.append(self.calculate_precision_at_k(true_ids, pred_ids, 10))
            aps.append(self.average_precision(true_ids, pred_ids))
            mrrs.append(self.calculate_reciprocal_rank(true_ids, pred_ids))

        mean_recall = sum(recalls) / len(recalls) if recalls else 0
        mean_precision_at_k = sum(precisions_k) / len(precisions_k) if precisions_k else 0
        mean_ap = sum(aps) / len(aps) if aps else 0
        mean_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
        return {
            "Mean Recall": mean_recall,
            "Precision@10": mean_precision_at_k,
            "Mean Average Precision": mean_ap,
            "Mean Reciprocal Rank": mean_mrr 
        }

evaluation_metrics = EvaluationMetrics()

@app.route('/evaluate_service', methods=['POST'])
def evaluate():
    data = request.json
    true_data = evaluation_metrics.load_data(data['true_data'])
    predictions = evaluation_metrics.load_predictions(data['predictions'])
    metrics = evaluation_metrics.calculate_metrics(true_data, predictions)
    calculate_reciprocal = evaluation_metrics.calculate_reciprocal_rank(true_data, predictions)
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True, port=9007)












•	بنية الـ main  لاستدعاء الخدمات التالية في تابع ال search و  غيرها ...

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
    clean_query = requests.post('http://localhost:9001/clean_query', json={"query": query}).json()['cleaned_query']
    
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
    dataset_name = 'clinical'  # You need to decide how to get the correct dataset
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
        query_Optimization.tfidf_queries(dataset_option,output_file)

        evaluate_service_url = "http://localhost:9007/evaluate_service"

        data_to_send = {
          "true_data": open(f"{dataset_option}/cleaned_queries.jsonl", 'r').read(),
          "predictions": open(f"{dataset_option}/tfidf_query_results.json", 'r').read()
                      }
        response = requests.post(evaluate_service_url, json=data_to_send)
        if response.status_code == 200:
           metrics = response.json()
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






5.	توصيف الطلبات الإضافية والخطوات الرئيسية لكل service :

•	الطلب الأول : Query refinement
1.	AutoComplete Service:
دخل هذه الخدمة هو مسار الـ Dataset و الـ   Queryيتم حساب الـ tf-idf للكويري بناء على ال tf-idf matrix الذي يتم بناءه على الملف المخزن فيه الكوريات وحساب  cosine_similarity   وإعادة النتائج 
class AutoCompleteService:
    def __init__(self, database):
        self.database = database
        self.autocomplete_dict = defaultdict(int)
        self.query_logs = []
        self.queries = []
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.base_dir, "databases", self.database)
        self.ensure_directories()
        self.load_data()
        self.load_common_queries()
        self.save_data()  # Save the data after loading common queries

    def ensure_directories(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def load_data(self):
        try:
            with open(os.path.join(self.data_path, "autocomplete_dict.json"), 'r') as f:
                self.autocomplete_dict = json.load(f)
            # Convert the loaded dict to defaultdict
            self.autocomplete_dict = defaultdict(int, self.autocomplete_dict)
        except (FileNotFoundError, json.JSONDecodeError):
            self.autocomplete_dict = defaultdict(int)
        print("Data loaded successfully")

    def load_common_queries(self):
        try:
            with open(os.path.join(self.data_path, "common_queries.txt"), 'r') as file:
                self.queries = [line.strip() for line in file.readlines()]
            for query in self.queries:
                terms = query.split()
                for term in terms:
                    self.autocomplete_dict[term] += 1
        except FileNotFoundError:
            print("common_queries.txt not found.")
        print("Common queries loaded successfully")

    def save_data(self):
        with open(os.path.join(self.data_path, "autocomplete_dict.json"), 'w') as f:
            # Convert defaultdict to a regular dict before saving
            json.dump(dict(self.autocomplete_dict), f)
        print("Data saved successfully")

    def autocomplete(self, query, dataset):
        corrected_query = str(TextBlob(query).correct())
        suggestions = []
        try:
            for term in self.autocomplete_dict.keys():
                if query in term:
                    suggestions.append(term)
            suggestions = sorted(suggestions, key=lambda x: self.autocomplete_dict[x], reverse=True)[:10]
            query_suggestions = [q for q in self.queries if query in q]
            return {
                "corrected_query": corrected_query,
                "autocomplete_suggestions": suggestions[:10],
                "query_suggestions": query_suggestions[:10]
            }
        except Exception as e:
            return {"error": str(e)}

@app.route('/autocomplete', methods=['POST'])
def process_query():
    try:
        data = request.json
        query = data['query']
        database = data['dataset']
        autocomplete_service = AutoCompleteService(database)
        result = autocomplete_service.autocomplete(query, database)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Failed to process autocomplete: {str(e)}")
        return jsonify({'error': 'Internal Server Error', 'details': str(e)}), 500


•	الطلب الثاني : Clustring

    def __init__(self, num_clusters=3):
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
    indexing = Indexing()
    tfidf_matrix = indexing.query_vectorize(tf_idf_file)
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



