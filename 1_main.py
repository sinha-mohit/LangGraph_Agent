import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

response = llm.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.2-Exp",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(response.choices[0].message)
# print(response)