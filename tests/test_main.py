from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app
from src.core.indexing import build_index

client = TestClient(app)

def setup_module(module):
    """
    Build the index before running the tests.
    """
    build_index()

@patch('src.core.aspirations.generation_model')
def test_recommend_with_aspirations(mock_generation_model):
    mock_response = MagicMock()
    mock_response.text = "independence, self-reliance, personal freedom"
    mock_generation_model.generate_content.return_value = mock_response
    user_conv = {
        "ref_user_id": 123,
        "messages_list": [
            {"screen_name": "user", "message": "I want to be more independent."}
        ]
    }
    response = client.post("/v1/recommend", json=user_conv)
    assert response.status_code == 200
    data = response.json()
    assert data["ref_user_id"] == 123
    assert "recommendations" in data
    assert "debug" in data
    # The order of tags might vary, so we check for set equality
    assert set(data["debug"]["top_aspirations"]) == set(["independence", "self-reliance", "personal freedom"])

@patch('src.core.aspirations.generation_model')
def test_recommend_without_aspirations(mock_generation_model):
    mock_response = MagicMock()
    mock_response.text = "other"
    mock_generation_model.generate_content.return_value = mock_response
    user_conv = {
        "ref_user_id": 123,
        "messages_list": [
            {"screen_name": "user", "message": "This is a test message."}
        ]
    }
    response = client.post("/v1/recommend", json=user_conv)
    assert response.status_code == 200
    data = response.json()
    assert data["ref_user_id"] == 123
    assert "recommendations" in data
    assert "debug" in data
    assert data["debug"]["top_aspirations"] == ["other"]

def test_recommend_empty_conversation():
    user_conv = {
        "ref_user_id": 123,
        "messages_list": []
    }
    response = client.post("/v1/recommend", json=user_conv)
    assert response.status_code == 200
    data = response.json()
    assert data["ref_user_id"] == 123
    assert data["recommendations"] == []
    assert data["debug"]["top_aspirations"] == []

def test_get_similar_posts():
    # Assuming there's a post with id 1131 in the index
    response = client.get("/v1/similar_posts/1131")
    assert response.status_code == 200
    data = response.json()
    assert data["post_id"] == 1131
    assert "similar_posts" in data

def test_get_similar_posts_not_found():
    response = client.get("/v1/similar_posts/999999")
    assert response.status_code == 200 # The API returns a 200 with an error message
    data = response.json()
    assert data["error"] == "Post not found."
