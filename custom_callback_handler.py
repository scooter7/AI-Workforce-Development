from langchain_community.callbacks import StreamlitCallbackHandler

class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    def __init__(self, parent_container):
        super().__init__(parent_container)
        self.final_response = None  # Store only the final response

    def on_llm_new_token(self, token: str, **kwargs):
        """Override to suppress intermediate tokens."""
        pass  # Do nothing to prevent intermediate outputs

    def on_chain_end(self, outputs, **kwargs):
        """Store only the final response from the agent."""
        self.final_response = outputs.get("text", None)
