import google.generativeai as genai
import os

# Set your API key (replace with your actual key or use environment variable)
genai.configure(api_key=os.getenv('GOOGLE_API_KEY', ""))

# List available models
for model in genai.list_models():
    print(f"Model: {model.name}")
    print(f"Display Name: {model.display_name}")
    print(f"Description: {model.description}")
    print("---")