from google import genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

print(GEMINI_API_KEY)
print(WEATHER_API_KEY)
print(GOOGLE_API_KEY)