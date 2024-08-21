import requests
import json
class Query_Optimization:

    def clean_relevant_query(self, dataset_name):
        with open(f"{dataset_name}/dev/qas.search.jsonl", "r") as f:
            queries = [json.loads(line) for line in f]

        with open(f"{dataset_name}/cleaned_queries.jsonl", "w") as f:
            for query in queries:
                response = requests.post('http://localhost:9001/clean_query', json={"query": query["query"],"datasetname":dataset_name})
                if response.status_code == 200:
                    clean_query = response.json()['cleaned_query']
                else:
                    clean_query = "Error in cleaning query"
                
                query_id = query["qid"]
                answer_pids = query["answer_pids"]
                cleaned_query_data = {
                    "qid": query_id,
                    "cleaned_query": clean_query,
                    "answer_pids": answer_pids
                }
                json.dump(cleaned_query_data, f)
                f.write("\n")

    def tfidf_queries(self, dataset_name, file_path):
        with open(f"{dataset_name}/cleaned_queries.jsonl", "r") as f:
            queries = [json.loads(line) for line in f]

        results = []
        for query in queries:
            query_id = query["qid"]
            print(query_id)
            clean_query = query["cleaned_query"]
            
            # Vectorize the query using the remote indexing service
            vectorize_response = requests.post('http://localhost:9002/vectorize_query', json={"query": clean_query, "file_path": file_path})
            if vectorize_response.status_code == 200:
                query_vector = vectorize_response.json()['query_vector']
            else:
                query_vector = []  # Handling failed vectorization

            # Matching the query using the remote matching service
            match_response = requests.post('http://localhost:9003/match', json={"query_vector": query_vector, "file_path": file_path})
            if match_response.status_code == 200:
                sorted_non_zero_indices = match_response.json()['top_10_indices']
            else:
                sorted_non_zero_indices = []  # Handling failed matching

            result = {
                "qid": query_id,
                "sorted_non_zero_indices": sorted_non_zero_indices
            }
            results.append(result)

        with open(f"{dataset_name}/tfidf_query_results.json", "w") as f:
            json.dump(results, f, indent=4)  

