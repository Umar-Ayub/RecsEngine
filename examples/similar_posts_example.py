import requests
import json
import sys

def get_similar_posts(post_id: int):
    """
    Get similar posts for a given post ID.
    """
    response = requests.get(f"http://localhost:8000/v1/similar_posts/{post_id}")

    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python examples/similar_posts_example.py <post_id>")
        sys.exit(1)

    post_id = int(sys.argv[1])
    get_similar_posts(post_id)
