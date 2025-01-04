from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .config import OllamaConfig

class LLMManager:
    """Manages interactions with Ollama LLM"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.llm = ChatOllama(
            model=config.model,
            temperature=config.temperature,
            format=config.format,
            base_url=config.base_url,
            timeout=config.timeout,
            context_window=config.context_window
        )
        self.output_parser = StrOutputParser()
    
    def create_chain(self, template: str, input_variables: list[str]):
        """Creates a processing chain with the given template"""
        prompt = PromptTemplate(
            template=template,
            input_variables=input_variables
        )
        return prompt | self.llm | self.output_parser
    
    def process(self, template: str, **kwargs):
        """Process a single prompt with given variables"""
        chain = self.create_chain(
            template=template,
            input_variables=list(kwargs.keys())
        )
        return chain.invoke(kwargs) 