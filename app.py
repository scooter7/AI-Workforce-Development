from typing import Callable, TypeVar
import os
import inspect
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
from streamlit_chat import message
from streamlit_pills import pills
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph
import shutil

load_dotenv()

# Set OpenAI API Key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Set environment variables from Streamlit secrets or .env
os.environ["LINKEDIN_EMAIL"] = st.secrets.get("LINKEDIN_EMAIL", "")
os.environ["LINKEDIN_PASS"] = st.secrets.get("LINKEDIN_PASS", "")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") or st.secrets.get("LANGCHAIN_TRACING_V2", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "")
os.environ["SERPER_API_KEY"] = st.secrets.get("SERPER_API_KEY", "")
os.environ["FIRECRAWL_API_KEY"] = st.secrets.get("FIRECRAWL_API_KEY", "")
os.environ["LINKEDIN_SEARCH"] = st.secrets.get("LINKEDIN_JOB_SEARCH", "")

# Page configuration
st.set_page_config(layout="wide")
st.title("Career Assistant - 👨‍💼")

streamlit_analytics.start_tracking()

# Setup directories and paths
temp_dir = "temp"
dummy_resume_path = os.path.abspath("dummy_resume.pdf")

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Add dummy resume if it does not exist
if not os.path.exists(dummy_resume_path):
    default_resume_path = "https://github.com/scooter7/AI-Workforce-Development/blob/0f531500fc4ede3862ff5232a566455cc549e226/dummy_resume.pdf"
    shutil.copy(default_resume_path, dummy_resume_path)

# Sidebar - File Upload
uploaded_document = st.sidebar.file_uploader("Upload Your Resume", type="pdf")

if not uploaded_document:
    uploaded_document = open(dummy_resume_path, "rb")
    st.sidebar.write("Using a dummy resume for demonstration purposes. ")
    st.sidebar.markdown(f"[View Dummy Resume]({'https://drive.google.com/file/d/1vTdtIPXEjqGyVgUgCO6HLiG9TSPcJ5eM/view?usp=sharing'})", unsafe_allow_html=True)
    
bytes_data = uploaded_document.read()

filepath = os.path.join(temp_dir, "resume.pdf")
with open(filepath, "wb") as f:
    f.write(bytes_data)

st.markdown("**Resume uploaded successfully!**")

# Define OpenAI Model settings
settings = {
    "model": "gpt-4o-mini",
    "model_provider": "openai",
    "temperature": 0.3,
}

# Create the agent flow
flow_graph = define_graph()
message_history = StreamlitChatMessageHistory()

# Initialize session state variables
if "active_option_index" not in st.session_state:
    st.session_state["active_option_index"] = None
if "interaction_history" not in st.session_state:
    st.session_state["interaction_history"] = []
if "response_history" not in st.session_state:
    st.session_state["response_history"] = ["Hello! How can I assist you today?"]
if "user_query_history" not in st.session_state:
    st.session_state["user_query_history"] = ["Hi there! 👋"]

# Containers for the chat interface
conversation_container = st.container()
input_section = st.container()

# Define functions used above
def initialize_callback_handler(main_container: DeltaGenerator):
    V = TypeVar("V")

    def wrap_function(func: Callable[..., V]) -> Callable[..., V]:
        context = get_script_run_ctx()

        def wrapped(*args, **kwargs) -> V:
            add_script_run_ctx(ctx=context)
            return func(*args, **kwargs)

        return wrapped

    streamlit_callback_instance = CustomStreamlitCallbackHandler(
        parent_container=main_container
    )

    for method_name, method in inspect.getmembers(
        streamlit_callback_instance, predicate=inspect.ismethod
    ):
        setattr(streamlit_callback_instance, method_name, wrap_function(method))

    return streamlit_callback_instance

def execute_chat_conversation(user_input, graph):
    callback_handler_instance = initialize_callback_handler(st.container())
    callback_handler = callback_handler_instance
    try:
        output = graph.invoke(
            {
                "messages": list(message_history.messages) + [user_input],
                "user_input": user_input,
                "config": settings,
                "callback": callback_handler,
            },
            {"recursion_limit": 30},
        )
        message_output = output.get("messages")[-1]
        messages_list = output.get("messages")
        message_history.clear()
        message_history.add_messages(messages_list)

    except Exception as exc:
        return ":( Sorry, Some error occurred. Can you please try again?"
    return message_output.content

# Clear Chat functionality
if st.button("Clear Chat"):
    st.session_state["user_query_history"] = []
    st.session_state["response_history"] = []
    message_history.clear()
    st.rerun()  # Refresh the app to reflect the cleared chat

# for tracking the query.
streamlit_analytics.start_tracking()

# Display chat interface
with input_section:
    options = [
        "Summarize my resume",
        "Create a career path visualization based on my skills and interests from my resume",
        "Show me jobs that fit my resume in Cleveland County, Oklahoma",
        "Analyze my resume and suggest a suitable job role and search for relevant job listings",
        "Generate a cover letter for my resume",
    ]
    icons = ["🔍", "🌐", "📝", "📈", "💼"]

    selected_query = pills(
        "Pick a question for query:",
        options,
        clearable=None,  # type: ignore
        icons=icons,
        index=st.session_state["active_option_index"],
        key="pills",
    )
    if selected_query:
        st.session_state["active_option_index"] = options.index(selected_query)

    # Display text input form
    with st.form(key="query_form", clear_on_submit=True):
        user_input_query = st.text_input(
            "Query:",
            value=(selected_query if selected_query else "Summarize my resume"),
            placeholder="📝 Write your query or select from the above",
            key="input",
        )
        submit_query_button = st.form_submit_button(label="Send")

    if submit_query_button:
        if not uploaded_document:
            st.error("Please upload your resume before submitting a query.")
        elif user_input_query:
            # Process the query as usual if resume is uploaded
            chat_output = execute_chat_conversation(user_input_query, flow_graph)
            st.session_state["user_query_history"].append(user_input_query)
            st.session_state["response_history"].append(chat_output)
            st.session_state["last_input"] = user_input_query  # Save the latest input
            st.session_state["active_option_index"] = None

# Display chat history
if st.session_state["response_history"]:
    with conversation_container:
        for i in range(len(st.session_state["response_history"])):
            message(
                st.session_state["user_query_history"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="fun-emoji",
            )
            message(
                st.session_state["response_history"][i],
                key=str(i),
                avatar_style="bottts",
            )

streamlit_analytics.stop_tracking()
