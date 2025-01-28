from typing import Any
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit.external.langchain.streamlit_callback_handler import (
    StreamlitCallbackHandler,
    LLMThought,
)
from langchain.schema import AgentAction


class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    def write_agent_name(self, name: str):
        """Suppress agent name display to prevent intermediate steps from showing."""
        pass  # Do nothing instead of writing agent name

    def on_thought(self, thought: LLMThought):
        """Suppress 'thinking' messages from showing in the UI."""
        pass  # Ignore intermediate steps

    def on_action(self, action: AgentAction):
        """Suppress intermediate actions from being displayed."""
        pass  # Prevent actions from rendering

    def on_result(self, result: Any):
        """Only process and store the final response."""
        self._parent_container.write(result)  # Ensure only the last message appears
