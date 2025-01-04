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

## Architecture Overview

### Graph-Based Design

The project uses a graph-based architecture to create flexible and composable AI workflows. This design allows for:

- Modular components that can be easily combined
- Clear separation of concerns
- Extensible processing pipelines
- State management between processing steps

#### Components

1. **Nodes**

   - Base class: `BaseNode`
   - Purpose: Individual processing units that perform specific tasks
   - Key features:
     - Input/output key management
     - State handling
     - Logging capabilities
   - Examples:
     - `ChatNode`: Handles LLM interactions
     - `LoggingNode`: Manages logging and debugging

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
from src.llm_manager import LLMManager
from src.graphs import ChatGraph

# Initialize components
config = OllamaConfig(
    model="llama3.2",
    temperature=0.7
)
llm_manager = LLMManager(config)

# Create chat graph
chat_graph = ChatGraph(
    llm_manager=llm_manager,
    template="Your prompt template here"
)

# Use the graph
response = chat_graph.chat("Your question here")
```

### Creating Custom Nodes

```python
from src.nodes import BaseNode

class CustomNode(BaseNode):
    def __init__(self, input_key="input", output_key="output"):
        super().__init__(input_key, output_key)

    def execute(self, state):
        # Process input from state
        input_data = state.get(self.input_key)
        # Your processing logic here
        result = process_data(input_data)
        # Update state with result
        state[self.output_key] = result
        return state
```

### Creating Custom Graphs

```python
from src.graphs import BaseGraph

class CustomGraph(BaseGraph):
    def __init__(self, nodes, edges, entry_point):
        super().__init__(nodes, edges, entry_point)

    def process(self, input_data):
        initial_state = {"input": input_data}
        result = self.execute(initial_state)
        return result.get("output")
```

## Advanced Features

1. **Retry Logic**

   - Automatic retries for failed LLM calls
   - Configurable retry attempts and delays
   - Response validation

2. **Logging**

   - Comprehensive logging of execution flow
   - Performance metrics
   - Debug information
   - Both console and file output

3. **State Management**
   - Persistent state across node execution
   - Data validation between nodes
   - Error state handling

## Running the Example

1. Make sure Ollama is running:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
```

2. Run the interactive chat example:

```bash
python examples/simple_chat.py
```

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
