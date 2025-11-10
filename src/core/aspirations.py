from typing import List
import google.generativeai as genai
from .config import GOOGLE_API_KEY
import os

generation_model = genai.GenerativeModel('gemini-2.5-flash')

def extract_aspirations(text: str) -> List[str]:
    """
    Uses Gemini to extract key themes and aspirations from the user's message as a list of tags.
    """
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    prompt = f"""
    Read the following user message and extract the key themes and aspirations as a comma-separated list of tags.
    For example, if the user talks about wanting to learn a new skill, the tags could be 'lifelong learning, personal growth, new hobbies'.
    Return only the comma-separated list of tags.

    Message: "{text}"
    """
    try:
        response = generation_model.generate_content(prompt)
        # Split the response text by comma and strip whitespace from each tag
        tags = [tag.strip() for tag in response.text.split(',') if tag.strip()]
        return tags
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return []
