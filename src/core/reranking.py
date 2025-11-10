from typing import List, Dict
from .models import Post
from .indexing import get_post_by_index, get_post_index_by_id
import json

def load_activity_data():
    """
    Load activity data from activity.json.
    """
    try:
        with open("data/activity.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def calculate_popularity(activity_data: List[Dict]) -> Dict[int, int]:
    """
    Calculate the popularity of each post based on the number of interactions.
    """
    popularity = {}
    for activity in activity_data:
        post_id = activity["post_id"]
        popularity[post_id] = popularity.get(post_id, 0) + 1
    return popularity

def rerank_and_filter(
    recommendations: List[Dict],
    aspirations: List[str],
    k: int,
    popularity: Dict[int, int],
) -> List[Dict]:
    """
    Rerank and filter the recommendations.
    """
    # Boost posts that align with the user's aspirations
    for rec in recommendations:
        post_index = get_post_index_by_id(rec["post_id"])
        post = get_post_by_index(post_index)
        for aspiration in aspirations:
            if aspiration in post.text.lower():
                rec["score"] *= 1.2  # Boost score by 20%

    # Add popularity boost
    for rec in recommendations:
        rec["score"] += popularity.get(rec["post_id"], 0) * 0.01

    # Sort by score
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    # Penalize duplicates and filter
    final_recommendations = []
    seen_post_ids = set()
    for rec in recommendations:
        if rec["post_id"] not in seen_post_ids:
            final_recommendations.append(rec)
            seen_post_ids.add(rec["post_id"])

    return final_recommendations[:k]
