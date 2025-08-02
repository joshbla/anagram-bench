"""Test OpenRouter API following their quickstart guide exactly."""
import requests
import json
import os
from dotenv import load_dotenv

# Load environment
load_dotenv('.env.local')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
print(f"API Key loaded: {bool(OPENROUTER_API_KEY)}")
if OPENROUTER_API_KEY:
    print(f"API Key starts with: {OPENROUTER_API_KEY[:10]}...")

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "anagram-benchmark",
    "X-Title": "Anagram Benchmark Test",
    "Content-Type": "application/json"
  },
  data=json.dumps({
    "model": "openai/gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)

print(f"\nStatus Code: {response.status_code}")
print(f"Response: {response.text}")