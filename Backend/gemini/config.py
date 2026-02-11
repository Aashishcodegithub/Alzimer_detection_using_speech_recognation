import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment (.env)")

genai.configure(api_key=GEMINI_API_KEY)


def get_gemini_model():
    """
    Return a configured Gemini model instance.
    If you change model name, change in one place here.
    """
    return genai.GenerativeModel("gemini-1.5-flash")
