import ollama
import os

# Ensure Ollama server is running and the model is pulled (e.g., ollama run llama3)
# No API key needed for local Ollama instance
# client = ollama.Client() # Use this if you need a persistent client instance, otherwise direct calls are fine

response = ollama.chat(
    model='llama3', # Replace with your desired Ollama model
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello, who won the world series in 2020?'},
    ]
)

print(response['message']['content'])
