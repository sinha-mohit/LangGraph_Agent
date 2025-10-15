import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Define the tools for the agent
def addFile(filename: str) -> str:
   """Create a new file in current directory"""
   if not os.path.exists(filename):
      with open(filename, "w") as f:
          pass
      print(f"File '{filename}' created.")
   else:
      print(f"File '{filename}' already exists.")


def addFolder(directory_name: str):
  """Create a new Directory in current directory"""
  if not os.path.exists(directory_name):
      os.mkdir(directory_name)
      print(f"Directory '{directory_name}' created.")
  else:
      print(f"Directory '{directory_name}' already exists.")


agent = create_react_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=[addFile, addFolder],
    prompt="You are a helpful assistant that helps users to manage files and directories in their current working directory. " \
    "You have access to two tools: addFile and addFolder. " \
    "Use these tools to create files and directories as per the user's request. " \
    "Always ensure to confirm the creation of files or directories, and handle cases where they already exist."
)

# Run the agents
response = agent.invoke(
    {"messages": [{"role": "user", "content": "create a new directory with name educosys"}]}
)

print(response)