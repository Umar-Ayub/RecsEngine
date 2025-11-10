from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .indexing import get_post_embedding_by_index

def calculate_aspirational_alignment(
    recommendations: List[Dict], aspirations: List[str]
) -> float:
    """
    Calculate the percentage of recommended posts that align with the user's aspirations.
    """
    if not recommendations or not aspirations:
        return 0.0

    aligned_posts = 0
    for rec in recommendations:
        post_text = rec["text"].lower()
        for aspiration in aspirations:
            if aspiration in post_text:
                aligned_posts += 1
                break
    return aligned_posts / len(recommendations)


def calculate_diversity(recommendations: List[Dict]) -> float:
    """
    Calculate the diversity of the recommendations as the average intra-list cosine distance.
    """
    if len(recommendations) < 2:
        return 0.0

    embeddings = []
    for rec in recommendations:
        embeddings.append(get_post_embedding_by_index(rec["post_id"]))

    if not embeddings:
        return 0.0
        
    embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings)
    
    # Get the upper triangle of the similarity matrix, excluding the diagonal
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    
    if upper_triangle_indices[0].size == 0:
        return 0.0

    # Calculate the average similarity
    average_similarity = np.mean(similarity_matrix[upper_triangle_indices])
    
    # Diversity is 1 - average similarity
    diversity = 1 - average_similarity
    
    return float(diversity)
