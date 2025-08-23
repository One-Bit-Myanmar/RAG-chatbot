import json
import os
import schedule
import time
from datetime import datetime
from main import RAG
from sentence_transformers import SentenceTransformer, util

# Initialize RAG
rag = RAG(search_key="YOUR_TAVILY_KEY")
rag.load_vectorstore()

# Semantic similarity model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load evaluation dataset
with open("eval.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Directory to save JSON logs
log_dir = "rag_eval_logs"
os.makedirs(log_dir, exist_ok=True)

# Compute cosine similarity
def semantic_similarity(expected, predicted):
    emb1 = embed_model.encode(expected, convert_to_tensor=True)
    emb2 = embed_model.encode(predicted, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())

# Run evaluation
def run_evaluation():
    logs = []

    for sample in eval_data:
        query = sample["query"]
        expected = sample["expected"]
        qid = sample["id"]

        predicted = rag.ask(user_id="eval", query=query)
        score = semantic_similarity(expected, predicted)
        status = "Good" if score >= 0.7 else "Bad"

        logs.append({
            "id": qid,
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "semantic_similarity": score,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })

    # Save JSON for monitoring
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(log_dir, f"eval_{timestamp_str}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"Evaluation completed and saved: {file_path}")

# Schedule evaluation every hour
schedule.every(1).hours.do(run_evaluation)

print("Monitoring pipeline started. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(10)
