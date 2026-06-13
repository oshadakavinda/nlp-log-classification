import joblib
from sentence_transformers import SentenceTransformer

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
_model_classification = None
_model_last_loaded = None

def get_bert_classifier():
    global _model_classification, _model_last_loaded
    model_path = os.path.join(BASE_DIR, "models/log_classifier.joblib")
    if not os.path.exists(model_path):
        return None
    
    try:
        mtime = os.path.getmtime(model_path)
        if _model_classification is None or _model_last_loaded != mtime:
            print(f"[BERT] Loading/Reloading ML classifier from {model_path} (mtime: {mtime})")
            _model_classification = joblib.load(model_path)
            _model_last_loaded = mtime
    except Exception as e:
        print(f"[BERT] Error loading model: {e}")
        # Fall back to existing loaded model if it failed (e.g., file is being written)
        if _model_classification is None:
            return None
            
    return _model_classification


def classify_with_bert_with_confidence(log_message):
    clf = get_bert_classifier()
    if clf is None:
        return "Unclassified", 0.0
        
    embeddings = model_embedding.encode([log_message])
    probabilities = clf.predict_proba(embeddings)[0]
    max_prob = float(max(probabilities))
    
    if max_prob < 0.5:
        return "Unclassified", max_prob
        
    predicted_label = clf.predict(embeddings)[0]
    return predicted_label, max_prob


def classify_with_bert(log_message):
    label, _ = classify_with_bert_with_confidence(log_message)
    return label


if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)
