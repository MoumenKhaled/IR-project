from flask import Flask, request, jsonify
import os
from textblob import TextBlob
from flask_cors import CORS
import json
from collections import defaultdict

app = Flask(__name__)
CORS(app)

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

if __name__ == '__main__':
    app.run(debug=True, port=9004)
