import json
from main import RAG
from sentence_transformers import SentenceTransformer, util

# Load eval dataset
with open("eval_dataset.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Initialize RAG
rag = RAG(search_key="YOUR_TAVILY_KEY")
rag.load_vectorstore()  

# Load sentence transformer for semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_score(expected, predicted):
    emb1 = model.encode(expected, convert_to_tensor=True)
    emb2 = model.encode(predicted, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())  # similarity between 0-1

results = []

for sample in eval_data:
    query = sample["query"]
    expected = sample["expected"]
    qid = sample["id"]
    answer = rag.ask(user_id="eval", query=query)

    score = semantic_score(expected, answer)
    
    result = {
        "id": qid,
        "query": query,
        "expected": expected,
        "predicted": answer,
        "score": score # 0 (not good) ,1 (good)
    }
    results.append(result)
    print(f"query ({qid}) completed.")


# Save results
with open("semantic_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Average score
avg_score = sum(r["score"] for r in results) / len(results)
print(f"\n[INFO] Average semantic similarity score: {avg_score:.4f}")
