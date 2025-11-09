from fastapi import FastAPI
from typing import List
import numpy as np

from .core.models import UserConv, Post
from .core.aspirations import extract_aspirations
from .core.embed import embed_texts
from .core.indexing import (
    build_index,
    search_index,
    get_post_by_index,
    get_post_index_by_id,
    get_post_embedding_by_index,
)

app = FastAPI()

@app.on_event("startup")
def startup_event():
    build_index()

@app.post("/v1/recommend")
def recommend(conv: UserConv, k: int = 5):
    """
    Recommend posts based on user conversation.
    """
    aspirations = []
    for m in conv.messages_list:
        if m.get("screen_name") != "StoryBot":
            aspirations.extend(extract_aspirations(m["message"]))
    
    top_aspirations = list(set(aspirations))

    if not top_aspirations or all(a == "other" for a in top_aspirations):
        context = [m["message"] for m in conv.messages_list if m.get("screen_name")!="StoryBot"][-3:]
        if not context:
            return {"ref_user_id": conv.ref_user_id, "recommendations": [], "debug": {"top_aspirations": top_aspirations}}
        u_vector = embed_texts([" ".join(context)]).mean(axis=0, keepdims=True)
    else:
        aspiration_vectors = embed_texts(top_aspirations)
        u_vector = np.mean(aspiration_vectors, axis=0, keepdims=True)

    distances, indices = search_index(u_vector, k)
    
    results = []
    if len(indices) > 0:
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = 1 - distances[0][i] # L2 distance to similarity
            results.append({"post_id": get_post_by_index(idx).post_id, "score": float(score)})

    return {"ref_user_id": conv.ref_user_id, "recommendations": results, "debug": {"top_aspirations": top_aspirations}}

@app.get("/v1/similar_posts/{post_id}")
def get_similar_posts(post_id: int, n: int = 5):
    """
    Get the n most similar posts to a given post.
    """
    try:
        post_index = get_post_index_by_id(post_id)
    except ValueError:
        return {"error": "Post not found."}

    post_embedding = get_post_embedding_by_index(post_index).reshape(1, -1)
    distances, indices = search_index(post_embedding, n + 1) # +1 to exclude the post itself

    results = []
    if len(indices) > 0:
        for i in range(1, len(indices[0])): # start from 1 to exclude the post itself
            idx = indices[0][i]
            score = 1 - distances[0][i] # L2 distance to similarity
            results.append({"post_id": get_post_by_index(idx).post_id, "score": float(score)})

    return {"post_id": post_id, "similar_posts": results}