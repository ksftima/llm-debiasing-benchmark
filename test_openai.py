from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Say hello!"}],
)
print(response.choices[0].message.content)
