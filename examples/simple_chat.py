import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OllamaConfig
from src.llm_manager import LLM_Manager
from src.graphs import Chat_Graph

def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
            logging.FileHandler('chat_log.txt')  # Output to file
        ]
    )

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = OllamaConfig(
        model="llama3.2",  # Make sure you have pulled this model with: ollama pull llama3.2
        temperature=0.7,  # Higher temperature for more creative responses
    )
    
    # Initialize LLM manager
    llm_manager = LLM_Manager(config)
    
    # Example template for chat
    chat_template = """You are a helpful AI assistant.
    Please answer the following question: {question}
    Provide a clear and concise response."""
    
    # Initialize chat graph
    chat_graph = Chat_Graph(llm_manager, chat_template)
    
    print("\nWelcome to the Chat Interface!")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("-" * 50)
    
    while True:
        # Get user input
        question = input("\nYou: ").strip()
        
        # Check for exit command
        if question.lower() in ['quit', 'exit', 'bye', 'goodbye', 'see you']:
            print("\nGoodbye!")
            break
        
        # Skip empty input
        if not question:
            continue
        
        try:
            logger.info("-" * 50)
            logger.info(f"Processing question: {question}")
            
            response = chat_graph.chat(question)
            
            # Print response in a user-friendly format
            print("\nAssistant:", response)
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print("\nError:", str(e))
        
        logger.info("-" * 50)

if __name__ == "__main__":
    main() 