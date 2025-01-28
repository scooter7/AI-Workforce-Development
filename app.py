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

# Styling for Streamlit app
st.markdown("""
<style>
/* General body styling */
body {
    font-family: 'Inter', sans-serif;
}

/* Sidebar styling (black background with white text) */
.st-emotion-cache-6qob1r.e1dbuyne8 {
    background-color: #121212 !important; /* Black sidebar background */
    color: #ffffff !important; /* White text */
}

.st-emotion-cache-6qob1r.e1dbuyne8 h2, 
.st-emotion-cache-6qob1r.e1dbuyne8 h3, 
.st-emotion-cache-6qob1r.e1dbuyne8 h4 {
    color: #f2a65a !important; /* Orange headings in sidebar */
}

/* Sidebar file uploader text */
.st-emotion-cache-6qob1r.e1dbuyne8 .stFileUploader label {
    color: #ffffff !important; /* White text for "Upload Your Resume" */
}

/* Main content area (white background with black text) */
.stApp {
    background-color: #ffffff !important; /* White main content background */
    color: #000000 !important; /* Black text */
}

/* Button styling */
.stButton button {
    background-color: #f2a65a !important; /* Orange button background */
    color: #000000 !important; /* Black button text */
    font-weight: bold !important;
    padding: 0.8rem 1.5rem !important;
    border-radius: 5px !important;
    transition: all 0.3s ease !important;
}

.stButton button:hover {
    background-color: #ffa94d !important; /* Lighter orange on hover */
    color: #ffffff !important; /* White button text on hover */
}

/* Input field styling */
.stTextInput input {
    background-color: #f9f9f9 !important; /* Light gray input background */
    color: #000000 !important; /* Black text in input */
    border: 1px solid #cccccc !important; /* Light gray border */
    border-radius: 5px !important;
    padding: 10px;
}

.stTextInput input:focus {
    outline: none;
    box-shadow: 0 0 5px #f2a65a !important; /* Orange focus effect */
}

/* Chat message styling */
.chat-container .message-user {
    background-color: #0a74da !important; /* Blue user message background */
    color: #ffffff !important; /* White user message text */
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 12px;
}

.chat-container .message-bot {
    background-color: #f9f9f9 !important; /* Light gray bot message background */
    color: #000000 !important; /* Black bot message text */
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

streamlit_analytics.start_tracking()

# Setup directories and paths
temp_dir = "temp"
dummy_resume_path = os.path.abspath("dummy_resume.pdf")

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Add dummy resume if it does not exist
if not os.path.exists(dummy_resume_path):
    default_resume_path = "path/to/your/dummy_resume.pdf"
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
    # Initialize the callback handler
    callback_handler_instance = initialize_callback_handler(st.container())

    try:
        # Invoke the agent and get response
        output = graph.invoke(
            {
                "messages": list(message_history.messages) + [user_input],
                "user_input": user_input,
                "config": settings,
                "callback": callback_handler_instance,  # Suppress intermediate steps
            },
            {"recursion_limit": 30},
        )

        # Extract only text content (prevent JSON serialization errors)
        if isinstance(output, str):
            message_output = output.strip()  # Direct response
        elif isinstance(output, dict) and "messages" in output:
            messages = [
                msg["content"] if isinstance(msg, dict) and "content" in msg else str(msg)
                for msg in output["messages"]
            ]
            message_output = messages[-1] if messages else "Error: No valid response received."
        else:
            message_output = "Error: Unexpected response format."

        # Ensure the chat history does NOT repeat responses
        if "last_input" in st.session_state and st.session_state["last_input"] == user_input:
            return  # Prevent duplicate entries

        # Convert HumanMessage objects to strings before storing them
        st.session_state["user_query_history"].append(str(user_input))
        st.session_state["response_history"].append(str(message_output))
        st.session_state["last_input"] = user_input  # Prevent duplicates

    except Exception as exc:
        return ":( Sorry, an error occurred. Please try again."

    # Return final response
    return message_output

# Clear Chat functionality
if st.button("Clear Chat"):
    st.session_state["user_query_history"] = []
    st.session_state["response_history"] = []
    st.session_state["last_input"] = None  # Prevent accidental duplicate queries
    message_history.clear()
    st.rerun()  # Refresh app to reflect cleared chat

# For tracking the query
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
            value=(selected_query if selected_query else ""),
            placeholder="📝 Write your query or select from the above",
            key="input",
        )
        submit_query_button = st.form_submit_button(label="Send")

    if submit_query_button:
        if not uploaded_document:
            st.error("Please upload your resume before submitting a query.")
        elif user_input_query:
            # Process the query
            chat_output = execute_chat_conversation(user_input_query, flow_graph)
            st.session_state["user_query_history"].append(user_input_query)
            st.session_state["response_history"].append(chat_output)
            st.session_state["last_input"] = user_input_query  # Save the latest input
            st.session_state["active_option_index"] = None

# Display chat history (only latest user query and final response)
if st.session_state["response_history"]:
    with conversation_container:
        for i in range(len(st.session_state["response_history"])):
            # Convert any non-string objects to strings
            user_message = str(st.session_state["user_query_history"][i])
            response_message = str(st.session_state["response_history"][i])

            # Display user query and final response (no intermediate steps)
            message(
                user_message,
                is_user=True,
                key=f"user_{i}",
                avatar_style="fun-emoji",
            )
            message(
                response_message,
                key=f"response_{i}",
                avatar_style="bottts",
            )

streamlit_analytics.stop_tracking()
