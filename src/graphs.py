"""
Graph implementations for chat workflows.
"""
from typing import Dict, Any, List, Tuple
import logging
from .nodes import Base_Node, Chat_Node, Logging_Node

class Base_Graph:
    """Base class for all graphs."""
    
    def __init__(self, nodes: List[Base_Node], edges: List[Tuple[Base_Node, Base_Node]], entry_point: Base_Node):
        self.nodes = nodes
        self.edges = self._create_edges(edges)
        self.entry_point = entry_point.node_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _create_edges(self, edges: List[Tuple[Base_Node, Base_Node]]) -> Dict[str, str]:
        """Create edge dictionary from list of node tuples."""
        return {from_node.node_name: to_node.node_name for from_node, to_node in edges}
    
    def _get_node_by_name(self, node_name: str) -> Base_Node:
        """Get node instance by name."""
        return next(node for node in self.nodes if node.node_name == node_name)
    
    def execute(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the graph workflow."""
        current_node_name = self.entry_point
        state = initial_state
        
        self.logger.info(f"Starting graph execution with initial state: {initial_state}")
        
        while current_node_name:
            current_node = self._get_node_by_name(current_node_name)
            try:
                self.logger.info(f"Executing node: {current_node_name}")
                state = current_node.execute(state)
                current_node_name = self.edges.get(current_node_name)
            except Exception as e:
                self.logger.error(f"Error in node {current_node_name}: {str(e)}")
                raise Exception(f"Error in node {current_node_name}: {str(e)}")
        
        self.logger.info("Graph execution completed")
        return state

class Chat_Graph(Base_Graph):
    """Graph implementation for chat workflows."""
    
    def __init__(self, llm_manager, template: str):
        # Create nodes
        chat_node = Chat_Node(
            llm_manager=llm_manager,
            template=template
        )
        
        logging_node = Logging_Node(
            log_keys=["question", "response", "execution_time"]
        )
        
        super().__init__(
            nodes=[chat_node, logging_node],
            edges=[(chat_node, logging_node)],
            entry_point=chat_node
        )
    
    def chat(self, question: str) -> str:
        """Process a chat question and return the response."""
        self.logger.info(f"Starting chat with question: {question}")
        state = {"question": question}
        result = self.execute(state)
        return result.get("response", "No response generated.") 