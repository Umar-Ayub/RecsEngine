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
from .core.reranking import rerank_and_filter, load_activity_data, calculate_popularity
from .core.metrics import calculate_aspirational_alignment, calculate_diversity
from .core.config import GOOGLE_API_KEY

app = FastAPI()

activity_data = []
popularity = {}

@app.on_event("startup")
def startup_event():
    build_index()
    global activity_data, popularity
    activity_data = load_activity_data()
    popularity = calculate_popularity(activity_data)


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

    distances, indices = search_index(u_vector, k * 2) # Get more recommendations to rerank
    
    initial_recommendations = []
    if len(indices) > 0:
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = 1 - distances[0][i]
            initial_recommendations.append({"post_id": int(idx), "score": float(score), "text": get_post_by_index(idx).text})

    final_recommendations = rerank_and_filter(initial_recommendations, top_aspirations, k, popularity)

    aspirational_alignment = calculate_aspirational_alignment(final_recommendations, top_aspirations)
    diversity = calculate_diversity(final_recommendations)

    return {
        "ref_user_id": conv.ref_user_id,
        "recommendations": final_recommendations,
        "debug": {
            "top_aspirations": top_aspirations,
            "metrics": {
                "aspirational_alignment": aspirational_alignment,
                "diversity": diversity,
            },
        },
    }

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
    distances, indices = search_index(post_embedding, n + 1)

    results = []
    if len(indices) > 0:
        for i in range(1, len(indices[0])):
            idx = indices[0][i]
            score = 1 - distances[0][i]
            results.append({"post_id": get_post_by_index(idx).post_id, "score": float(score)})

    return {"post_id": post_id, "similar_posts": results}