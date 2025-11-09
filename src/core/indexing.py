import faiss
import numpy as np
from typing import List
from .models import Post
from .embed import embed_texts
import json

POSTS: List[Post] = []
POST_EMB: np.ndarray = None
FAISS_INDEX: faiss.Index = None

def build_index():
    """
    Load posts from discussions.json and build FAISS index.
    """
    global POSTS, POST_EMB, FAISS_INDEX
    try:
        with open("data/discussions.json", "r") as f:
            discussions = json.load(f)
        
        posts_to_index = []
        for discussion in discussions:
            if discussion["messages_list"]:
                post_message = discussion["messages_list"][0]
                if not post_message["reported_or_removed"]:
                    posts_to_index.append(Post(post_id=post_message["post_id"], text=post_message["text"]))

        POSTS = posts_to_index
        if POSTS:
            POST_EMB = embed_texts([p.text for p in POSTS])
            d = POST_EMB.shape[1]
            FAISS_INDEX = faiss.IndexFlatL2(d)
            FAISS_INDEX.add(POST_EMB)

    except FileNotFoundError:
        print("Warning: data/discussions.json not found. Starting with an empty index.")
        POSTS = []
        POST_EMB = np.array([])
        FAISS_INDEX = None

def search_index(vector: np.ndarray, k: int):
    """
    Search the FAISS index for the k nearest neighbors.
    """
    if FAISS_INDEX is None:
        return [], []
    return FAISS_INDEX.search(vector, k)

def get_post_by_index(index: int) -> Post:
    """
    Get a post by its index in the POSTS list.
    """
    return POSTS[index]

def get_post_index_by_id(post_id: int) -> int:
    """
    Get the index of a post in the POSTS list by its ID.
    """
    for i, post in enumerate(POSTS):
        if post.post_id == post_id:
            return i
    raise ValueError("Post not found")

def get_post_embedding_by_index(index: int) -> np.ndarray:
    """
    Get the embedding of a post by its index.
    """
    return POST_EMB[index]
