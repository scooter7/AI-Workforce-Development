from typing import Any
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit.external.langchain.streamlit_callback_handler import (
    StreamlitCallbackHandler,
    LLMThought,
)
from langchain.schema import AgentAction

class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    def write_agent_name(self, name: str):
        """Prevent the agent name from being displayed."""
        pass  # Do nothing

    def on_thought(self, thought: LLMThought):
        """Suppress 'thinking' messages from showing in the UI."""
        pass  # Ignore intermediate steps

    def on_action(self, action: AgentAction):
        """Prevent intermediate agent steps from being displayed."""
        pass  # Do nothing

    def on_result(self, result: Any):
        """Ensure only the final response is displayed."""
        self._parent_container.write(result)  # Only show the final result
