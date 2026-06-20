import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('HF_API_KEY_1')

client = OpenAI(base_url='https://router.huggingface.co/v1', api_key=api_key)

prompt = """You are a document classifier. 
Given this OCR text, identify what type of document this is.

Return ONLY this JSON:
{
  "detected_type": "<AadhaarCard|Passport|DrivingLicence|VoterID|PANCard|DeathCertificate|ClaimantStatementForm|Unknown>",
  "confidence": <0.0 to 1.0>,
  "reason": "<one line why>"
}

OCR Text:
GOVERNMENT OF INDIA. DEATH CERTIFICATE. Name: John Doe. Date of Death: 12-10-2022.
"""

resp = client.chat.completions.create(
    model='Qwen/Qwen2.5-72B-Instruct',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)

print(resp.choices[0].message.content)
