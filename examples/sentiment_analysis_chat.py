"""
Enhanced chat example using custom nodes and graph for complex processing.
"""
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OllamaConfig
from src.llm_manager import LLM_Manager
from src.nodes import Base_Node, Logging_Node, Chat_Node
from src.graphs import Base_Graph

class Sentiment_Analysis_Node(Base_Node):
    """Node for analyzing sentiment of user input."""
    
    def __init__(self, llm_manager, input_key="question", output_key="sentiment"):
        super().__init__(input_key=input_key, output_key=output_key)
        self.llm_manager = llm_manager
        self.template = """Analyze the sentiment of the following text and respond with EXACTLY ONE WORD from these options: positive/neutral/negative.

        Guidelines:
        - Use 'neutral' for factual statements, questions, or statements without clear emotion
        - Use 'positive' for statements expressing happiness, excitement, gratitude, or other positive emotions
        - Use 'negative' for statements expressing anger, sadness, frustration, or other negative emotions
        
        Text: {question}
        
        Respond with just one word (positive/neutral/negative):"""
    
    def _validate_sentiment(self, sentiment: str) -> str:
        """Validate and clean the sentiment response."""
        valid_sentiments = {"positive", "neutral", "negative"}
        cleaned = sentiment.strip().lower()
        
        if cleaned not in valid_sentiments:
            self.logger.warning(f"Invalid sentiment detected: {cleaned}, defaulting to neutral")
            return "neutral"
        
        return cleaned
    
    def execute(self, state):
        question = state.get(self.input_key)
        if not question:
            raise ValueError(f"Input key '{self.input_key}' not found in state")
            
        response = self.llm_manager.process(template=self.template, question=question)
        sentiment = self._validate_sentiment(response)
        
        self.logger.info(f"Detected sentiment: {sentiment} for text: {question}")
        
        # Update state with validated sentiment
        state[self.output_key] = sentiment
        state['sentiment'] = sentiment
        
        return state

class Context_Enhanced_Chat_Node(Chat_Node):
    """Enhanced chat node that considers sentiment in responses."""
    
    def __init__(self, llm_manager):
        template = """You are an empathetic AI assistant that adapts its tone based on user sentiment.
        User's message: {question}
        Detected sentiment: {sentiment}
        
        Please provide a response that matches the emotional tone of the user while being helpful and constructive.
        If the sentiment is negative, be extra supportive and understanding.
        If the sentiment is positive, match their enthusiasm.
        If the sentiment is neutral, maintain a balanced and professional tone.
        If the sentiment is not detected, respond neutrally."""
        
        super().__init__(
            llm_manager=llm_manager,
            template=template,
            input_key="question",  # We'll handle sentiment separately
            output_key="response"
        )
    
    def execute(self, state):
        # Ensure we have both question and sentiment before proceeding
        if "sentiment" not in state:
            raise ValueError("Sentiment analysis must be performed before chat response")
        if "question" not in state:
            raise ValueError("Question must be provided")
            
        # Create input dictionary with both required variables
        input_dict = {
            "question": state["question"],
            "sentiment": state["sentiment"]
        }
        
        # Update state with our input dictionary
        state.update(input_dict)
        return super().execute(state)

class Enhanced_Chat_Graph(Base_Graph):
    """Advanced chat graph with sentiment analysis and enhanced responses."""
    
    def __init__(self, llm_manager):
        # Create nodes
        sentiment_node = Sentiment_Analysis_Node(llm_manager)
        chat_node = Context_Enhanced_Chat_Node(llm_manager)
        logging_node = Logging_Node(
            log_keys=["question", "sentiment", "response", "execution_time"]
        )
        
        # Define graph structure
        super().__init__(
            nodes=[sentiment_node, chat_node, logging_node],
            edges=[
                (sentiment_node, chat_node),
                (chat_node, logging_node)
            ],
            entry_point=sentiment_node
        )
    
    def chat(self, question: str) -> str:
        """Process a chat question with sentiment analysis and return the response."""
        state = {"question": question}
        result = self.execute(state)
        return result.get("response", "No response generated.")

def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('chat_log.txt')
        ]
    )

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = OllamaConfig(
        model="llama3.2",
        temperature=0.7,
    )
    
    # Initialize LLM manager
    llm_manager = LLM_Manager(config)
    
    # Initialize enhanced chat graph
    chat_graph = Enhanced_Chat_Graph(llm_manager)
    
    print("\nWelcome to the Enhanced Chat Interface!")
    print("This chat bot analyzes sentiment and adapts its responses accordingly.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("-" * 50)
    
    while True:
        # Get user input
        question = input("\nYou: ").strip()
        
        # Check for exit command
        if question.lower() in ['quit', 'exit', 'bye', 'goodbye', 'see you']:
            print("\nGoodbye! 👋")
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