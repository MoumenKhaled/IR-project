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
