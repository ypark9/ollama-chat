"""
Node classes for chat graph implementation.
"""
from typing import Any, Dict
import time
import logging
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Base_Node:
    """Base class for all nodes in the graph."""
    
    def __init__(self, input_key: str = None, output_key: str = None):
        self.input_key = input_key
        self.output_key = output_key
        self.node_name = self.__class__.__name__
        self.logger = logging.getLogger(self.node_name)
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node's operation on the input state."""
        raise NotImplementedError("Subclasses must implement execute method")

class Logging_Node(Base_Node):
    """Node for logging state information."""
    
    def __init__(self, log_keys: list = None):
        super().__init__()
        self.log_keys = log_keys or []
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Log the specified state information."""
        self.logger.info("Current state:")
        for key in self.log_keys:
            if key in state:
                self.logger.info(f"{key}: {state[key]}")
        return state

class Chat_Node(Base_Node):
    """Node for handling chat interactions using Ollama."""
    
    def __init__(self, llm_manager, template: str, input_key: str = "question", output_key: str = "response",
                 max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(input_key=input_key, output_key=output_key)
        self.llm_manager = llm_manager
        self.template = template
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _clean_response(self, response: str) -> str:
        """Clean the response by removing excessive whitespace and normalizing JSON."""
        try:
            # Try to parse as JSON first
            cleaned = json.loads(response)
            return json.dumps(cleaned, ensure_ascii=False)
        except json.JSONDecodeError:
            # If not valid JSON, clean up the text
            # Remove multiple newlines
            cleaned = re.sub(r'\n\s*\n', '\n', response)
            # Remove trailing/leading whitespace
            cleaned = cleaned.strip()
            # Normalize spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned
    
    def _is_valid_response(self, response: str) -> bool:
        """Check if the response is valid and non-empty."""
        if not response:
            return False
        
        try:
            # If it's JSON, check if it's empty
            parsed = json.loads(response)
            return bool(parsed)  # Returns False for empty dict/list/etc
        except json.JSONDecodeError:
            # If it's not JSON, check if it's just whitespace
            return bool(response.strip())
    
    def _prepare_template_variables(self, state: Dict[str, Any]) -> dict:
        """Prepare variables for template substitution."""
        # Extract all variables needed by the template using regex
        template_vars = re.findall(r'\{(\w+)\}', self.template)
        template_dict = {}
        
        # Fill in all required variables from state
        for var in template_vars:
            if var not in state:
                raise ValueError(f"Required template variable '{var}' not found in state")
            template_dict[var] = state[var]
        
        return template_dict
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the chat input and generate a response with retry logic."""
        self.logger.info(f"Processing question: {state.get(self.input_key)}")
        
        # Prepare template variables
        template_vars = self._prepare_template_variables(state)
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.llm_manager.process(
                    template=self.template,
                    **template_vars  # Pass all template variables
                )
                execution_time = time.time() - start_time
                
                # Clean the response
                cleaned_response = self._clean_response(response)
                
                # Check if response is valid
                if self._is_valid_response(cleaned_response):
                    self.logger.info(f"Response generated in {execution_time:.2f} seconds (attempt {attempt + 1})")
                    self.logger.info(f"Response: {cleaned_response}")
                    
                    state[self.output_key] = cleaned_response
                    state["execution_time"] = execution_time
                    return state
                else:
                    self.logger.warning(f"Empty or invalid response on attempt {attempt + 1}, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Error in ChatNode (attempt {attempt + 1}): {last_error}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
        
        # If we get here, all retries failed
        error_msg = f"Failed to get valid response after {self.max_retries} attempts"
        if last_error:
            error_msg += f": {last_error}"
        raise Exception(error_msg) 