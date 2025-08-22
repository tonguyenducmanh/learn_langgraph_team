import streamlit as st
import uuid
import os
from typing import List, TypedDict
import requests
import json

from dotenv import load_dotenv
from batch_processor import render_batch_processor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- API Tools ---
API_URL = "http://0.0.0.0:8000"

@tool
def generate_image_tool(query: str):
    """Generates an image based on a query."""
    try:
        response = requests.post(f"{API_URL}/generate-image/", params={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error calling generate_image API: {e}"

@tool
def generate_story_tool(title: str, author: str, pages: List[dict]):
    """Generates a story book."""
    try:
        book_data = {"title": title, "author": author, "pages": pages}
        response = requests.post(f"{API_URL}/generate-story/", json=book_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error calling generate_story API: {e}"

tools = [generate_image_tool, generate_story_tool]


# --- Helper Functions ---

def generate_title(llm, messages: List[BaseMessage]) -> str:
    """Generates a title for a conversation using the LLM."""
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages if isinstance(msg, (HumanMessage, AIMessage))])
    if not conversation_text:
        return "Cuộc trò chuyện mới"
    
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

# Set up the language model and bind tools
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

# Define the state for our graph
class AgentState(TypedDict):
    messages: List[BaseMessage]

# Define the nodes for our graph
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"

# Define the graph
workflow = StateGraph(AgentState)
tool_node = ToolNode(tools)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")

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

    threads = list(metadata_collection.find({}, sort=[("_id", -1)]))
    for thread in threads:
        if st.sidebar.button(thread['title'], key=thread['thread_id'], use_container_width=True):
            st.session_state.thread_id = thread['thread_id']
            checkpoint_tuple = memory.get_tuple({"configurable": {"thread_id": thread['thread_id']}})
            if checkpoint_tuple:
                st.session_state.messages = checkpoint_tuple.checkpoint['channel_values']['messages']
            else:
                st.session_state.messages = [SystemMessage(content="Bạn là một AI agent chuyên sáng tác truyện cổ tích. Bạn có khả năng tạo ra hình ảnh minh họa cho câu chuyện và tạo ra một cuốn sách truyện hoàn chỉnh. Dựa vào yêu cầu của người dùng, hãy chủ động sử dụng các công cụ `generate_image_tool` để vẽ tranh hoặc `generate_story_tool` để tạo sách. Hãy hỏi người dùng nếu bạn cần thêm thông tin để hoàn thành câu chuyện.")]
            st.rerun()

    # --- Main UI ---
    st.title("AI Agent Kể Chuyện Cổ Tích")

    # Render the batch processing UI from the separate module
    render_batch_processor(llm)

    # --- Interactive Chat UI ---
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state["messages"] = [
            SystemMessage(content="Bạn là một AI agent chuyên sáng tác truyện cổ tích. Bạn có khả năng tạo ra hình ảnh minh họa cho câu chuyện và tạo ra một cuốn sách truyện hoàn chỉnh. Dựa vào yêu cầu của người dùng, hãy chủ động sử dụng các công cụ `generate_image_tool` để vẽ tranh hoặc `generate_story_tool` để tạo sách. Hãy hỏi người dùng nếu bạn cần thêm thông tin để hoàn thành câu chuyện.")
        ]

    for message in st.session_state.messages:
        if isinstance(message, (HumanMessage, AIMessage)):
            with st.chat_message(message.type):
                st.markdown(message.content)
        elif isinstance(message, ToolMessage):
            with st.chat_message("tool"):
                st.markdown(f"Tool Output: {message.content}")


    if prompt := st.chat_input("Bạn muốn nghe chuyện gì?"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("human"):
            st.markdown(prompt)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        graph_input = {"messages": st.session_state.messages}

        with st.spinner("AI đang suy nghĩ..."):
            # Stream the graph execution
            for event in app.stream(graph_input, config=config):
                for key, value in event.items():
                    if key == "agent":
                        if value['messages'][-1].tool_calls:
                            st.session_state.messages.append(value['messages'][-1])
                        else:
                             st.session_state.messages.append(value['messages'][-1])
                    elif key == "tools":
                        st.session_state.messages.append(value['messages'][-1])


            is_new_thread = metadata_collection.count_documents({"thread_id": st.session_state.thread_id}) == 0
            if is_new_thread:
                title = generate_title(llm, st.session_state.messages)
                metadata_collection.insert_one({"thread_id": st.session_state.thread_id, "title": title})
        st.rerun()
