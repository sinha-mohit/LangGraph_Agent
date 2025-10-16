from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

checkpointer = InMemorySaver()

agent = create_react_agent(
   model="groq:llama-3.3-70b-versatile", 
   tools=[], 
   checkpointer=checkpointer,
   prompt="You are a helpful assistant" 
)


config = {"configurable": {"thread_id": "1"}}

first_response = agent.invoke(
   {"messages": [{"role": "user", "content": "who is modi in one line"}]},
   config = config
)

second_response = agent.invoke(
   {"messages": [{"role": "user", "content": "when was he born and in which city?"}]},
   config = config
)

print(first_response['messages'][-1].content)
print('-------------')
print(second_response['messages'][-1].content)


while True:
   try:
      user_input = input("You: ")
      if user_input.lower() in ["exit", "quit"]:
         print("Exiting the chat. Goodbye!")
         break

      response = agent.invoke(
         {"messages": [{"role": "user", "content": user_input}]},
         config = config
      )
      print("Assistant:", response['messages'][-1].content)

   except KeyboardInterrupt:
      print("\nExiting the chat. Goodbye!")
      break