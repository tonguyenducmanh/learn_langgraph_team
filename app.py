import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import List, TypedDict
import os
import uuid

# Load environment variables
load_dotenv()

# Set up the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Define the state for our graph
class AgentState(TypedDict):
    messages: List[BaseMessage]


# Define the nodes for our graph
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm.invoke(messages)
    # Return a dictionary with the updated messages
    return {"messages": messages + [response]}

# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# Set up the checkpoint saver
mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DATABASE_NAME", "langgraph_db") # Use default if not set

if not mongodb_uri:
    st.error("MONGODB_URI environment variable not set. Please create a .env file with MONGODB_URI.")
    st.stop()

with MongoDBSaver.from_conn_string(mongodb_uri, db_name=db_name, collection_name="chat_threads") as memory:
    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    # Streamlit UI
    st.title("AI Agent Kể Chuyện Cổ Tích")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            SystemMessage(content="Bạn là một người kể chuyện cổ tích AI. Hãy bắt đầu một câu chuyện và tương tác với người dùng.")
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if isinstance(message, (HumanMessage, AIMessage)):
            with st.chat_message(message.type):
                st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("Bạn muốn nghe chuyện gì?"):
        # Add user message to session state and display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("human"):
            st.markdown(prompt)

        # Initialize a unique thread_id for the session if it doesn't exist
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())

        # Define a unique thread for the conversation
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Prepare the input for the graph
        graph_input = {"messages": st.session_state.messages}

        # Invoke the graph
        with st.spinner("AI đang suy nghĩ..."):
            # The graph will automatically save the state to MongoDB
            final_state = app.invoke(graph_input, config=config) # type: ignore
            # The final state is a dictionary
            ai_response = final_state["messages"][-1]

        # Add AI response to session state and display it
        st.session_state.messages.append(ai_response)
        with st.chat_message("ai"):
            st.markdown(ai_response.content)
