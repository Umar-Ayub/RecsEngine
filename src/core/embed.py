from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embeds a list of texts using a sentence-transformer model.
    """
    return embedding_model.encode(texts)
