from dotenv import load_dotenv
import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel




load_dotenv()

async def run_agent():
    client = MultiServerMCPClient(
        {
            "elasticsearch-mcp-server": {
                "url": "http://localhost:8089/mcp",
                "transport": "streamable_http" # stream-http, server-sent-events, stdio
            }
        }
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    tools = await client.get_tools()
    agent = create_react_agent(
        llm, 
        tools, 
        checkpointer=InMemorySaver(),
        prompt="You are a helpful assistant." 
    )

    
    config = {"configurable": {"thread_id": "1"}}

    print("Type 'exit' to quit.")
    while True:
        user_input = input("User: ").strip()
        
        # skip if empty
        if not user_input:
            continue
        
        if user_input.lower() == "exit":
            break
        
        input_message = {"role": "user", "content": user_input}
        
        async for response in agent.astream(
            {"messages": [input_message]},
            stream_mode="values",
            config=config
        ):  

            print("-------------------")
            resp = response["messages"][-1]
            if hasattr(resp, "content") and isinstance(resp.content, list):
                for item in resp.content:
                    if isinstance(item, dict) and item.get("type") == "thinking":
                        print("Thinking:", item["thinking"])
                    elif isinstance(item, str):
                        print("Final Response:", item)
            else:
                print(resp)
        



if __name__ == "__main__":
   asyncio.run(run_agent())
