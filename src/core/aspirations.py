from typing import List
import google.generativeai as genai
from .config import GOOGLE_API_KEY, ASPIRATION_CATEGORIES
import os

generation_model = genai.GenerativeModel('gemini-2.5-flash')

def extract_aspirations(text: str) -> List[str]:
    """
    Uses Gemini to classify the user's message into one of the predefined aspiration categories.
    """
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    prompt = f"""
    Given the following user message, classify it into one of the following categories:
    {', '.join(ASPIRATION_CATEGORIES)}

    Message: "{text}"

    Return only the category name, or "other" if it doesn't fit into any of the categories.
    """
    try:
        response = generation_model.generate_content(prompt)
        return [response.text.strip()]
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return []
