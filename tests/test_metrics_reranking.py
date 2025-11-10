# ... (existing imports) ...
import copy
from src.core.models import Post
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.core.metrics import calculate_aspirational_alignment, calculate_diversity
from src.core.reranking import load_activity_data, calculate_popularity, rerank_and_filter

# Mock data for testing
MOCK_POSTS_DATA = {
    1: Post(post_id=1, text="This post is about independence and freedom."),
    2: Post(post_id=2, text="A guide to financial independence."),
    3: Post(post_id=3, text="Travel tips for your next adventure."),
    4: Post(post_id=4, text="Healthy eating habits."),
    5: Post(post_id=5, text="Another post about freedom."),
}

MOCK_EMBEDDINGS_DATA = {
    1: np.array([0.1, 0.2, 0.3]),
    2: np.array([0.1, 0.2, 0.3]),
    3: np.array([0.4, 0.5, 0.6]),
    4: np.array([0.7, 0.8, 0.9]),
    5: np.array([0.1, 0.2, 0.3]),
}

# Create mock POSTS and POST_EMB based on MOCK_POSTS_DATA
MOCK_POSTS_LIST = list(MOCK_POSTS_DATA.values())
MOCK_POST_EMBEDDINGS = [None] * (max(MOCK_POSTS_DATA.keys()) + 1)

for post_id, embedding in MOCK_EMBEDDINGS_DATA.items():
    MOCK_POST_EMBEDDINGS[post_id] = embedding

@patch('src.core.indexing.POST_EMB', new=MOCK_POST_EMBEDDINGS)
def test_calculate_aspirational_alignment():
    # Test case 1: Empty recommendations
    assert calculate_aspirational_alignment([], ["independence"]) == 0.0

    # Test case 2: Empty aspirations
    recommendations = [{"post_id": 1, "text": "independence"}, {"post_id": 2, "text": "travel"}]
    assert calculate_aspirational_alignment(recommendations, []) == 0.0

    # Test case 3: Some alignment
    recommendations = [
        {"post_id": 1, "text": "This post is about independence."},
        {"post_id": 2, "text": "A guide to financial freedom."},
        {"post_id": 3, "text": "Travel tips for your next adventure."},
    ]
    aspirations = ["independence", "freedom"]
    # Post 1 aligns with "independence", Post 2 aligns with "freedom"
    assert calculate_aspirational_alignment(recommendations, aspirations) == 2/3

    # Test case 4: No alignment
    recommendations = [
        {"post_id": 3, "text": "Travel tips for your next adventure."},
        {"post_id": 4, "text": "Healthy eating habits."},
    ]
    aspirations = ["independence", "freedom"]
    assert calculate_aspirational_alignment(recommendations, aspirations) == 0.0

    # Test case 5: Full alignment
    recommendations = [
        {"post_id": 1, "text": "This post is about independence."},
        {"post_id": 2, "text": "A guide to financial freedom."},
    ]
    aspirations = ["independence", "freedom"]
    assert calculate_aspirational_alignment(recommendations, aspirations) == 1.0

@patch('src.core.indexing.POST_EMB', new=MOCK_POST_EMBEDDINGS)
def test_calculate_diversity():
    # Test case 1: Less than 2 recommendations
    assert calculate_diversity([]) == 0.0
    assert calculate_diversity([{"post_id": 1}]) == 0.0

    # Test case 2: All recommendations are the same (low diversity)
    recommendations_same = [
        {"post_id": 1}, {"post_id": 2}, {"post_id": 5} # All have the same embedding
    ]
    # Cosine similarity of identical vectors is 1. Diversity = 1 - 1 = 0
    assert calculate_diversity(recommendations_same) == pytest.approx(0.0)

    # Test case 3: Recommendations are different (high diversity)
    recommendations_diverse = [
        {"post_id": 1}, {"post_id": 3}, {"post_id": 4}
    ]
    # Embeddings:
    # 1: [0.1, 0.2, 0.3]
    # 3: [0.4, 0.5, 0.6]
    # 4: [0.7, 0.8, 0.9]
    # Calculate expected diversity manually or with a helper if complex
    # For now, just check it's greater than 0
    diversity = calculate_diversity(recommendations_diverse)
    assert diversity > 0.0
    assert diversity <= 1.0 # Diversity is between 0 and 1

    # Test case 4: Two recommendations with different embeddings
    recommendations_two_diverse = [
        {"post_id": 1}, {"post_id": 3}
    ]
    # Embedding 1: [0.1, 0.2, 0.3]
    # Embedding 3: [0.4, 0.5, 0.6]
    # Cosine similarity between [0.1, 0.2, 0.3] and [0.4, 0.5, 0.6]
    # dot product = 0.1*0.4 + 0.2*0.5 + 0.3*0.6 = 0.04 + 0.10 + 0.18 = 0.32
    # mag1 = sqrt(0.1^2 + 0.2^2 + 0.3^2) = sqrt(0.01 + 0.04 + 0.09) = sqrt(0.14) approx 0.374
    # mag3 = sqrt(0.4^2 + 0.5^2 + 0.6^2) = sqrt(0.16 + 0.25 + 0.36) = sqrt(0.77) approx 0.877
    # similarity = 0.32 / (0.374 * 0.877) = 0.32 / 0.328 = 0.975
    # diversity = 1 - 0.975 = 0.025
    expected_similarity = np.dot(MOCK_EMBEDDINGS_DATA[1], MOCK_EMBEDDINGS_DATA[3]) / \
                          (np.linalg.norm(MOCK_EMBEDDINGS_DATA[1]) * np.linalg.norm(MOCK_EMBEDDINGS_DATA[3]))
    expected_diversity = 1 - expected_similarity
    assert calculate_diversity(recommendations_two_diverse) == pytest.approx(expected_diversity)


@patch('builtins.open', new_callable=MagicMock)
@patch('json.load')
def test_load_activity_data(mock_json_load, mock_open):
    # Test case 1: File exists and contains data
    mock_json_load.return_value = [{"post_id": 1, "user_id": 101}]
    assert load_activity_data() == [{"post_id": 1, "user_id": 101}]
    mock_open.assert_called_once_with("data/activity.json", "r")

    # Test case 2: File not found
    mock_open.side_effect = FileNotFoundError
    assert load_activity_data() == []
    mock_open.assert_called_with("data/activity.json", "r") # Still called

def test_calculate_popularity():
    # Test case 1: Empty activity data
    assert calculate_popularity([]) == {}

    # Test case 2: Some activity data with default weights (all 'read')
    activity_data = [
        {"post_id": 1, "ref_user_id": 101, "activity_type": "read"},
        {"post_id": 2, "ref_user_id": 101, "activity_type": "read"},
        {"post_id": 1, "ref_user_id": 102, "activity_type": "read"},
    ]
    # Default read weight is 0.5
    expected_popularity = {1: 1.0, 2: 0.5}
    assert calculate_popularity(activity_data) == expected_popularity

    # Test case 3: Single activity with default weight
    activity_data_single = [{"post_id": 5, "ref_user_id": 200, "activity_type": "read"}]
    assert calculate_popularity(activity_data_single) == {5: 0.5}

def test_calculate_popularity_weighted():
    # Test case 1: Mixed activity types with default weights
    activity_data = [
        {"post_id": 1, "ref_user_id": 101, "activity_type": "read"},
        {"post_id": 2, "ref_user_id": 101, "activity_type": "commented"},
        {"post_id": 1, "ref_user_id": 102, "activity_type": "created"},
        {"post_id": 3, "ref_user_id": 103, "activity_type": "read"},
        {"post_id": 2, "ref_user_id": 102, "activity_type": "read"},
    ]
    # Default weights: commented=3.0, created=2.5, read=0.5
    # Post 1: read (0.5) + created (2.5) = 3.0
    # Post 2: commented (3.0) + read (0.5) = 3.5
    # Post 3: read (0.5) = 0.5
    expected_popularity = {1: 3.0, 2: 3.5, 3: 0.5}
    assert calculate_popularity(activity_data) == expected_popularity

    # Test case 2: Custom weights
    custom_weights = {"read": 1.0, "commented": 5.0, "created": 0.1}
    activity_data_custom_weights = [
        {"post_id": 1, "ref_user_id": 101, "activity_type": "read"},
        {"post_id": 1, "ref_user_id": 102, "activity_type": "commented"},
        {"post_id": 2, "ref_user_id": 103, "activity_type": "created"},
    ]
    # Post 1: read (1.0) + commented (5.0) = 6.0
    # Post 2: created (0.1) = 0.1
    expected_popularity_custom = {1: 6.0, 2: 0.1}
    assert calculate_popularity(activity_data_custom_weights, weights=custom_weights) == expected_popularity_custom

    # Test case 3: Activity type not in weights, should use default 0.5 (for 'read')
    activity_data_unknown_type = [
        {"post_id": 1, "ref_user_id": 101, "activity_type": "unknown"},
    ]
    expected_popularity_unknown = {1: 0.5}
    assert calculate_popularity(activity_data_unknown_type) == expected_popularity_unknown



