import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HF_API_KEY_1") or os.getenv("OPENAI_API_KEY")
print(f"Using API Key: {api_key[:5] if api_key else 'None'}...")

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=api_key
)

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    import traceback
    print("Error Type:", type(e))
    print("Error:", e)
    traceback.print_exc()
