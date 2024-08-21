from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import nltk
from nltk.corpus import stopwords
import string
import datetime

app = Flask(__name__)

class DataPreprocessing:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def lemmatize_words(self, words):
        return [self.lemmatizer.lemmatize(word) for word in words]
    
    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def remove_phonetic_notation(self, text):
        pattern = r'\[[^\]]+\]'
        return re.sub(pattern, '', text)

    def tokenize_text(self, text):
        return re.split(r'\s+', text.strip())

    def remove_urls(self, tokens):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return [token for token in tokens if not re.match(url_pattern, token)]

    def process_and_standardize_dates_brackets(self, tokens):
        processed_tokens = []
        for token in tokens:
            try:
                date = datetime.datetime.strptime(token, '%Y-%m-%d').date()
                processed_tokens.append(date.strftime('%Y-%m-%d'))
            except ValueError:
                token = re.sub(r'[\[\]]', '', token)
                processed_tokens.append(token)
        return processed_tokens

    def remove_punctuation(self, tokens):
        translator = str.maketrans('', '', string.punctuation)
        return [token.translate(translator) for token in tokens]

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def convert_to_lowercase(self, tokens):
        return [token.lower() for token in tokens]

    def join_tokens(self, tokens):
        return ' '.join(tokens)

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
        dataset_terms = []

        with open(data_set_path, 'r', encoding='utf-8') as file:
            data = file.read().split("\n")
            for i, row in enumerate(data):
 
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

if __name__ == '__main__':
    app.run(debug=True, port=9001)