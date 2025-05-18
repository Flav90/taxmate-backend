
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import json

# Load tax laws from JSON
with open("taxmate_tz_laws.json", "r", encoding="utf-8") as f:
    structured_laws = json.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [s['text'] for s in structured_laws]
embeddings = model.encode(texts, convert_to_tensor=True)

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    message: str

@app.post("/ask")
async def ask_tax_law(req: QueryRequest):
    query_embedding = model.encode(req.message, convert_to_tensor=True)
    top_match = util.semantic_search(query_embedding, embeddings, top_k=1)[0][0]
    match = structured_laws[top_match['corpus_id']]
    return {
        "section": f"{match['act']} – Section {match['section']}: {match['title']}",
        "answer": match['text']
    }
