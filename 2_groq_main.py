import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
response = model.invoke("who is modi")
print(response.content)