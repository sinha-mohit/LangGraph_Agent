"""
Production-Ready Streamlit Chatbot with LangGraph
Features:
- Proper error handling
- Loading states
- Optimized session state management
- Robust streaming implementation
- Production-ready configuration
- Chat history management (ChatGPT-style)
- Session persistence
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        margin: 10px 0;
    }
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables with proper defaults"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()
    
    if "llm" not in st.session_state:
        try:
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.7,
                max_tokens=2048
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            st.stop()
    
    if "config" not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": "main_thread"}}
    
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = create_react_agent(
                st.session_state.llm, 
                tools=[], 
                checkpointer=st.session_state.checkpointer,
                prompt="""You are a helpful AI assistant. Provide accurate, helpful, and concise responses. 
                If you don't know something, say so rather than making up information."""
            )
        except Exception as e:
            st.error(f"Failed to create agent: {str(e)}")
            st.stop()

def display_chat_history():
    """Display the chat history with proper formatting"""
    for role, message in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(message)

def stream_agent_response(user_input: str) -> str:
    """
    Stream the agent's response with proper error handling and loading states
    
    Args:
        user_input: The user's input message
        
    Returns:
        The complete assistant response
    """
    assistant_response = ""
    
    try:
        with st.chat_message("assistant"):
            # Create placeholder for streaming response
            message_placeholder = st.empty()
            
            # Add loading indicator
            with message_placeholder.container():
                st.markdown("🤖 Thinking...")
                time.sleep(0.5)  # Brief pause for UX
            
            # Stream the response
            for event in st.session_state.agent.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                st.session_state.config
            ):
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict) and "messages" in node_output:
                        # Get the latest message content
                        messages = node_output["messages"]
                        if messages and hasattr(messages[-1], 'content'):
                            new_text = messages[-1].content
                            if new_text and new_text != assistant_response:
                                # Only update if we have new content
                                assistant_response = new_text
                                message_placeholder.markdown(assistant_response)
            
            # If no streaming content was received, show a fallback
            if not assistant_response:
                assistant_response = "I apologize, but I couldn't generate a response. Please try again."
                message_placeholder.markdown(assistant_response)
                
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        assistant_response = "I encountered an error while processing your request. Please try again."
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    
    return assistant_response

def create_new_chat():
    """Create a new chat session"""
    # Save current chat if it has messages
    if st.session_state.messages:
        save_current_chat()
    
    # Generate new chat ID
    new_chat_id = str(uuid.uuid4())
    chat_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Create new session
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = []
    st.session_state.config = {"configurable": {"thread_id": new_chat_id}}
    
    # Add to chat sessions
    st.session_state.chat_sessions[new_chat_id] = {
        "title": chat_title,
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    
    st.rerun()

def save_current_chat():
    """Save the current chat session"""
    if st.session_state.current_chat_id and st.session_state.messages:
        st.session_state.chat_sessions[st.session_state.current_chat_id] = {
            "title": st.session_state.chat_sessions.get(st.session_state.current_chat_id, {}).get("title", f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
            "messages": st.session_state.messages.copy(),
            "created_at": st.session_state.chat_sessions.get(st.session_state.current_chat_id, {}).get("created_at", datetime.now().isoformat()),
            "last_updated": datetime.now().isoformat()
        }

def load_chat_session(chat_id: str):
    """Load a specific chat session"""
    if chat_id in st.session_state.chat_sessions:
        # Save current chat before switching
        if st.session_state.messages:
            save_current_chat()
        
        # Load the selected chat
        session = st.session_state.chat_sessions[chat_id]
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = session["messages"].copy()
        st.session_state.config = {"configurable": {"thread_id": chat_id}}
        st.rerun()

def delete_chat_session(chat_id: str):
    """Delete a chat session"""
    if chat_id in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[chat_id]
        
        # If we're deleting the current chat, create a new one
        if st.session_state.current_chat_id == chat_id:
            create_new_chat()
        else:
            st.rerun()

def rename_chat_session(chat_id: str, new_title: str):
    """Rename a chat session"""
    if chat_id in st.session_state.chat_sessions:
        st.session_state.chat_sessions[chat_id]["title"] = new_title
        st.session_state.chat_sessions[chat_id]["last_updated"] = datetime.now().isoformat()
        st.rerun()

def clear_chat_history():
    """Clear the chat history and reset the conversation"""
    st.session_state.messages = []
    st.session_state.config = {"configurable": {"thread_id": f"thread_{int(time.time())}"}}
    st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Create initial chat if none exists
    if not st.session_state.current_chat_id:
        create_new_chat()
    
    # Sidebar for controls and chat history
    with st.sidebar:
        st.title("🤖 Chatbot")
        
        # New Chat Button
        if st.button("➕ New Chat", type="primary", use_container_width=True):
            create_new_chat()
        
        st.markdown("---")
        
        # Chat History Section
        st.markdown("### 💬 Chat History")
        
        if st.session_state.chat_sessions:
            # Sort sessions by last updated (most recent first)
            sorted_sessions = sorted(
                st.session_state.chat_sessions.items(),
                key=lambda x: x[1]["last_updated"],
                reverse=True
            )
            
            for chat_id, session in sorted_sessions:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Highlight current chat
                    if chat_id == st.session_state.current_chat_id:
                        st.button(
                            f"📌 {session['title']}",
                            key=f"chat_{chat_id}",
                            use_container_width=True,
                            type="primary"
                        )
                    else:
                        if st.button(
                            session['title'],
                            key=f"chat_{chat_id}",
                            use_container_width=True
                        ):
                            load_chat_session(chat_id)
                
                with col2:
                    # Delete button for each chat
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Delete this chat"):
                        delete_chat_session(chat_id)
                
                # Show last message preview
                if session['messages']:
                    last_message = session['messages'][-1][1]
                    preview = last_message[:50] + "..." if len(last_message) > 50 else last_message
                    st.caption(f"💬 {preview}")
                
                st.markdown("---")
        else:
            st.info("No chat history yet. Start a conversation!")
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info("Using Gemini 2.0 Flash")
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        - Click "New Chat" to start a fresh conversation
        - Click on any chat in history to continue it
        - Use the delete button to remove old chats
        """)
    
    # Main chat interface
    st.title("🤖 AI Chatbot")
    
    # Show current chat title
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chat_sessions:
        current_title = st.session_state.chat_sessions[st.session_state.current_chat_id]["title"]
        st.markdown(f"**Current Chat:** {current_title}")
    
    st.markdown("Ask me anything! I'm here to help.")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to session state
        st.session_state.messages.append(("user", prompt))
        
        # Generate and display assistant response
        with st.spinner("Generating response..."):
            assistant_response = stream_agent_response(prompt)
        
        # Add assistant response to session state
        st.session_state.messages.append(("assistant", assistant_response))
        
        # Auto-save the chat after each interaction
        save_current_chat()

if __name__ == "__main__":
    main()