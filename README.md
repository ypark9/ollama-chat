# Ollama Project Example

This project demonstrates how to use Ollama with LangChain for building AI applications using a graph-based architecture.

## Prerequisites

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the required model:

```bash
ollama pull llama3.3
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `src/`
  - `config.py`: Configuration management
  - `llm_manager.py`: LLM interaction management
  - `nodes.py`: Node implementations for the graph
  - `graphs.py`: Graph implementations and execution logic
- `examples/`
  - `simple_chat.py`: Interactive chat example using graph architecture
  - `sentiment_analysis_chat.py`: Enhanced chat example with sentiment analysis

## Architecture Overview

### Graph-Based Design

The project uses a graph-based architecture to create flexible and composable AI workflows. This design allows for:

- Modular components that can be easily combined
- Clear separation of concerns
- Extensible processing pipelines
- State management between processing steps

#### Components

1. **Nodes**

   - Base class: `Base_Node`
   - Purpose: Individual processing units that perform specific tasks
   - Key features:
     - Input/output key management
     - State handling
     - Logging capabilities
   - Examples:
     - `Chat_Node`: Handles LLM interactions
     - `Logging_Node`: Manages logging and debugging
     - `Sentiment_Analysis_Node`: Analyzes text sentiment
     - `Context_Enhanced_Chat_Node`: Provides context-aware responses

2. **Edges**

   - Purpose: Define the flow of data between nodes
   - Implementation: Directed connections between nodes
   - Usage: Determine the execution order and data flow

3. **Graphs**
   - Base class: `BaseGraph`
   - Purpose: Orchestrate the execution of nodes
   - Features:
     - State management across nodes
     - Error handling
     - Execution flow control
   - Examples:
     - `ChatGraph`: Implements a chat workflow

### Error Handling and Reliability

The architecture includes several reliability features:

- Retry logic for failed LLM calls
- Response validation
- Comprehensive logging
- Error propagation and handling

## Using the Graph Architecture

### Creating a Simple Chat Workflow

```python
from src.config import OllamaConfig
from src.llm_manager import LLM_Manager
from src.graphs import Chat_Graph

# Initialize components
config = OllamaConfig(
    model="llama3.2",
    temperature=0.7
)
llm_manager = LLM_Manager(config)

# Create chat graph
chat_graph = Chat_Graph(
    llm_manager=llm_manager,
    template="Your prompt template here"
)

# Use the graph
response = chat_graph.chat("Your question here")
```

### Creating a Sentiment-Aware Chat System

```python
from src.graphs import Base_Graph
from src.nodes import Sentiment_Analysis_Node, Context_Enhanced_Chat_Node, Logging_Node

class Enhanced_Chat_Graph(Base_Graph):
    def __init__(self, llm_manager):
        # Create nodes for sentiment analysis pipeline
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

# Initialize and use the enhanced chat
chat_graph = Enhanced_Chat_Graph(llm_manager)
response = chat_graph.chat("I'm feeling great today!")
```

### Advanced Features

1. **Sentiment Analysis**

   - Automatic sentiment detection (positive/neutral/negative)
   - Handles factual statements and questions appropriately
   - Robust validation and error handling
   - Sentiment-aware response generation

2. **Context-Enhanced Responses**

   - Adapts tone based on detected sentiment
   - Provides empathetic and appropriate responses
   - Handles multiple input variables
   - Maintains conversation context

3. **Retry Logic**

   - Automatic retries for failed LLM calls
   - Configurable retry attempts and delays
   - Response validation

## Running the Examples

1. Simple chat example:

```bash
python examples/simple_chat.py
```

2. Sentiment analysis chat:

```bash
python examples/sentiment_analysis_chat.py
```

The sentiment analysis chat provides:

- Automatic emotion detection
- Context-aware responses
- Empathetic interaction
- Comprehensive logging

## Troubleshooting

1. **Model not found error**: Make sure you've pulled the model:

   ```bash
   ollama pull llama3.2
   ```

2. **Connection error**: Ensure Ollama is running:

   ```bash
   curl http://localhost:11434/api/tags
   ```

3. **Empty or invalid responses**: The system will automatically retry with the configured retry logic

4. **Process killed error**: This might happen if:
   - The model is too large for your system's memory
   - Ollama process is not running properly
     Try restarting Ollama or using a smaller model.

## Configuration

You can modify the Ollama configuration in `config.py` or when creating an `OllamaConfig` instance:

```python
config = OllamaConfig(
    model="llama3.2",
    temperature=0.7,
    format="json",
    base_url="http://localhost:11434"
)
```

## Future Extensions

The graph-based architecture can be extended to support:

1. Parallel processing of nodes
2. Conditional execution paths
3. Dynamic graph construction
4. Integration with other AI models and services
5. Custom node types for specific use cases
