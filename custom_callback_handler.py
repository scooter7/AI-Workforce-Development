from typing import Any
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit.external.langchain.streamlit_callback_handler import (
    StreamlitCallbackHandler,
    LLMThought,
)
from langchain.schema import AgentAction, AgentFinish


class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    def __init__(self, parent_container):
        super().__init__(parent_container)
        self.final_response = None  # Store only the final response

    def write_agent_name(self, name: str):
        """Suppress writing agent name to prevent agent steps from showing."""
        pass  # Do nothing to prevent unnecessary agent messages

    def on_agent_action(self, action: AgentAction, **kwargs):
        """Suppress intermediate agent steps."""
        pass  # Do nothing to hide the agent's thought process

    def on_llm_new_token(self, token: str, **kwargs):
        """Suppress intermediate token streaming."""
        pass  # Prevents intermediate token streaming

    def on_chain_end(self, outputs, **kwargs):
        """Store the final response instead of intermediate steps."""
        if isinstance(outputs, dict):
            self.final_response = outputs.get("output", "")  # Adjusted key name

    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Capture the final agent response."""
        self.final_response = finish.return_values.get("output", "")
