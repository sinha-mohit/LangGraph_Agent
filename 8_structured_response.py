# Structured Response
# Examples - 
# Mail - sub, body
# Travel itinerary - Day1, day2, hotel, budget
# Code generation - language, code
# Health report - condition, medicine recommendation


from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Define the structured response format using Pydantic
class MailResponse(BaseModel):
   subject: str
   body: str


agent = create_react_agent(
   model="groq:llama-3.3-70b-versatile", 
   tools=[], 
   response_format = MailResponse 
)


config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
   {"messages": [{"role": "user", "content": "write a mail applying leave for travel"}]},
   config 
)
print(response)
print("------------------------------")
print(response["structured_response"])


print("------------------------------")
print(response["structured_response"].subject)


print("------------------------------")
print(response["structured_response"].body)
