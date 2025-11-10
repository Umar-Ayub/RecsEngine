import requests
import json
import sys

def get_recommendations(user_id: int):
    """
    Get recommendations for a user.
    """
    with open("data/conversations.json", "r") as f:
        conversations = json.load(f)

    user_conv = None
    for conv in conversations:
        if conv["ref_user_id"] == user_id:
            user_conv = conv
            break

    if not user_conv:
        print(f"No conversation found for user {user_id}")
        return

    response = requests.post("http://localhost:8000/v1/recommend", json=user_conv)

    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python examples/recommend_example.py <user_id>")
        sys.exit(1)

    user_id = int(sys.argv[1])
    get_recommendations(user_id)
