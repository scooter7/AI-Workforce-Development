from langchain_core.callbacks import BaseCallbackHandler

class CustomStreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.final_response = None  # Store only the final response

    def on_llm_new_token(self, token: str, **kwargs):
        """Override to suppress intermediate tokens."""
        pass  # Do nothing to prevent intermediate outputs

    def on_chain_end(self, outputs, **kwargs):
        """Store only the final response from the agent."""
        self.final_response = outputs.get("text", None)
