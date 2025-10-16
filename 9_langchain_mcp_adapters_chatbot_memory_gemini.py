from dotenv import load_dotenv
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

import os


load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


async def run_agent():
    client = MultiServerMCPClient(
        {
            "github": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-github"
                ],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN
                },
                "transport": "stdio"
            },

            "filesystem": {
               "command": "npx",
               "args": [
                   "-y",
                   "@modelcontextprotocol/server-filesystem",
                   "/Users/mohit/Documents/Software-Development/LangGraph_Agent/"
               ],
               "transport":"stdio"
           }

        }
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", include_thoughts=True)
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    print("Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break
        input_message = {"role": "user", "content": user_input}
        async for response in agent.astream(
            {"messages": [input_message]},
            stream_mode="values",
            config=config
        ):
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
