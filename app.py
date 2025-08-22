import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import List, TypedDict
import os
import uuid

# --- Helper Functions ---

def generate_title(llm, messages: List[BaseMessage]) -> str:
    """Generates a title for a conversation using the LLM."""
    # Format the conversation for the prompt
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    prompt = (
        "Hãy tạo một tiêu đề ngắn gọn (dưới 10 từ) tóm tắt nội dung cuộc trò chuyện sau. "
        "Chỉ trả về nội dung tiêu đề, không thêm bất kỳ lời giải thích nào.\n\n"
        f"Cuộc trò chuyện:\n{conversation_text}"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return "Cuộc trò chuyện mới"

# --- Main Application ---

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
    return {"messages": messages + [response]}

# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# Set up the checkpoint saver
mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DATABASE_NAME", "langgraph_db")

if not mongodb_uri:
    st.error("MONGODB_URI environment variable not set. Please create a .env file with MONGODB_URI.")
    st.stop()

with MongoDBSaver.from_conn_string(mongodb_uri, db_name=db_name, collection_name="chat_threads") as memory:
    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    # Get the pymongo database object to manage metadata
    db = memory.client[db_name]
    metadata_collection = db["chat_metadata"]

    # --- Sidebar for Chat History ---
    st.sidebar.title("Lịch sử trò chuyện")
    if st.sidebar.button("➕ Trò chuyện mới", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # Display list of past conversations
    threads = list(metadata_collection.find({}, sort=[("_id", -1)]))
    for thread in threads:
        if st.sidebar.button(thread['title'], key=thread['thread_id'], use_container_width=True):
            st.session_state.thread_id = thread['thread_id']
            # Load the history for the selected thread
            checkpoint_tuple = memory.get_tuple({"configurable": {"thread_id": thread['thread_id']}})
            if checkpoint_tuple:
                st.session_state.messages = checkpoint_tuple.checkpoint['channel_values']['messages']
            else:
                # Fallback for empty thread
                st.session_state.messages = [SystemMessage(content="Bạn là một người kể chuyện cổ tích AI.")]
            st.rerun()

    # --- Main Chat UI ---
    st.title("AI Agent Kể Chuyện Cổ Tích")

    # Initialize session state if it's a new session
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state["messages"] = [
            SystemMessage(content="Bạn là một người kể chuyện cổ tích AI. Hãy bắt đầu một câu chuyện và tương tác với người dùng.")
        ]

    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, (HumanMessage, AIMessage)):
            with st.chat_message(message.type):
                st.markdown(message.content)

    # React to user input
    if prompt := st.chat_input("Bạn muốn nghe chuyện gì?"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("human"):
            st.markdown(prompt)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        graph_input = {"messages": st.session_state.messages}

        with st.spinner("AI đang suy nghĩ..."):
            # Invoke the graph
            final_state = app.invoke(graph_input, config=config) # type: ignore
            ai_response = final_state["messages"][-1]
            st.session_state.messages.append(ai_response)

            # Check if a title needs to be generated (first user interaction)
            is_new_thread = metadata_collection.count_documents({"thread_id": st.session_state.thread_id}) == 0
            if is_new_thread:
                title = generate_title(llm, st.session_state.messages)
                metadata_collection.insert_one({"thread_id": st.session_state.thread_id, "title": title})

        st.rerun()
