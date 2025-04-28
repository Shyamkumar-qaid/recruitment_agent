"""
Centralized LLM Service for managing interactions with various LLM providers.

This module provides a unified interface for interacting with different LLM providers
including OpenRouter, Hugging Face, and Ollama. It handles authentication, model selection,
and provides a consistent API for text generation across all providers.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import json

# Import LangChain components
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

# Configure logging
logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: str
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)

class LLMService:
    """
    Centralized service for interacting with various LLM providers.
    
    This service provides a unified interface for text generation, chat completion,
    and other LLM operations across different providers including Ollama, OpenAI,
    OpenRouter, and Hugging Face.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM service with the specified configuration.
        
        Args:
            config: LLM configuration including provider, model, and API key
        """
        self.config = config or self._get_default_config()
        self.llm = self._initialize_llm()
        logger.info(f"LLM Service initialized with provider: {self.config.provider}, model: {self.config.model_name}")
    
    def _get_default_config(self) -> LLMConfig:
        """
        Get default LLM configuration from environment variables.
        
        Returns:
            LLMConfig with default settings
        """
        # Default to Ollama with phi3 model
        return LLMConfig(
            provider=LLMProvider.OLLAMA,
            model_name=os.getenv("OLLAMA_MODEL", "phi3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2
        )
    
    def _initialize_llm(self) -> BaseLLM:
        """
        Initialize the appropriate LLM based on the configuration.
        
        Returns:
            Configured LLM instance
        """
        provider = self.config.provider
        
        try:
            if provider == LLMProvider.OLLAMA:
                return self._initialize_ollama()
            elif provider == LLMProvider.OPENAI:
                return self._initialize_openai()
            elif provider == LLMProvider.OPENROUTER:
                return self._initialize_openrouter()
            elif provider == LLMProvider.HUGGINGFACE:
                return self._initialize_huggingface()
            else:
                logger.warning(f"Unsupported provider: {provider}. Falling back to Ollama.")
                return self._initialize_ollama()
        except Exception as e:
            logger.error(f"Error initializing LLM for provider {provider}: {str(e)}")
            logger.warning("Falling back to Ollama")
            # Fall back to Ollama if there's an error
            self.config.provider = LLMProvider.OLLAMA
            self.config.model_name = os.getenv("OLLAMA_MODEL", "phi3")
            self.config.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return self._initialize_ollama()
    
    def _initialize_ollama(self) -> BaseLLM:
        """Initialize Ollama LLM"""
        base_url = self.config.base_url or "http://localhost:11434"
        return Ollama(
            model=self.config.model_name,
            base_url=base_url,
            temperature=self.config.temperature,
            **self.config.additional_params
        )
    
    def _initialize_openai(self) -> BaseLLM:
        """Initialize OpenAI LLM"""
        try:
            from langchain_openai import ChatOpenAI
            
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required")
            
            # Set API key in environment
            os.environ["OPENAI_API_KEY"] = self.config.api_key
            
            return ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.additional_params
            )
        except ImportError:
            logger.error("langchain_openai package not installed. Please install it to use OpenAI.")
            raise
    
    def _initialize_openrouter(self) -> BaseLLM:
        """Initialize OpenRouter LLM"""
        try:
            from langchain_openai import ChatOpenAI
            
            if not self.config.api_key:
                raise ValueError("OpenRouter API key is required")
            
            # Set API key in environment for both OpenRouter and OpenAI
            # (CrewAI uses OpenAI's API for OpenRouter)
            os.environ["OPENROUTER_API_KEY"] = self.config.api_key
            os.environ["OPENAI_API_KEY"] = self.config.api_key
            
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=self.config.api_key,
                **self.config.additional_params
            )
        except ImportError:
            logger.error("langchain_openai package not installed. Please install it to use OpenRouter.")
            raise
    
    def _initialize_huggingface(self) -> BaseLLM:
        """Initialize Hugging Face LLM"""
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            
            if not self.config.api_key:
                raise ValueError("Hugging Face API token is required")
            
            return HuggingFaceEndpoint(
                endpoint_url=f"https://api-inference.huggingface.co/models/{self.config.model_name}",
                huggingfacehub_api_token=self.config.api_key,
                task="text-generation",
                temperature=self.config.temperature,
                max_length=self.config.max_tokens or 512,
                **self.config.additional_params
            )
        except ImportError:
            logger.error("langchain_huggingface package not installed. Please install it to use Hugging Face.")
            raise
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Generated text response
        """
        try:
            return self.llm.invoke(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_prompt_template(self, template: str, input_variables: Dict[str, Any], **kwargs) -> str:
        """
        Run a prompt template with the given input variables.
        
        Args:
            template: The prompt template string
            input_variables: Dictionary of variables to fill in the template
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Generated text response
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm
        return chain.invoke(input_variables, **kwargs)
    
    def structured_output(self, 
                         template: str, 
                         input_variables: Dict[str, Any], 
                         output_parser: BaseOutputParser,
                         **kwargs) -> Any:
        """
        Generate structured output using a prompt template and output parser.
        
        Args:
            template: The prompt template string
            input_variables: Dictionary of variables to fill in the template
            output_parser: Parser to convert LLM output to structured format
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Structured output from the parser
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | output_parser
        return chain.invoke(input_variables, **kwargs)
    
    def parse_json_output(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON output from LLM with fallback strategies.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        import re
        import json
        
        if not text:
            return None
            
        # Strategy 1: Try to parse the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Strategy 2: Look for JSON-like structure with regex
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
            
        # Strategy 3: Look for JSON with triple backticks (common in LLM outputs)
        try:
            code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_blocks:
                for block in code_blocks:
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
            
        # Strategy 4: Try to fix common JSON errors and parse again
        try:
            # Replace single quotes with double quotes (common LLM mistake)
            fixed_text = re.sub(r"'([^']*)':\s*", r'"\1": ', text)
            # Fix boolean values
            fixed_text = re.sub(r':\s*True', ': true', fixed_text)
            fixed_text = re.sub(r':\s*False', ': false', fixed_text)
            # Fix trailing commas
            fixed_text = re.sub(r',\s*}', '}', fixed_text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            
            # Try to find and parse JSON in the fixed text
            json_match = re.search(r'\{.*\}', fixed_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
            
        return None
    
    def update_config(self, new_config: LLMConfig) -> None:
        """
        Update the LLM configuration and reinitialize the LLM.
        
        Args:
            new_config: New LLM configuration
        """
        self.config = new_config
        self.llm = self._initialize_llm()
        logger.info(f"LLM Service updated with provider: {self.config.provider}, model: {self.config.model_name}")

# Factory function to create LLM service from model config dict
def create_llm_service(model_config: Dict[str, Any] = None) -> LLMService:
    """
    Create an LLM service from a model configuration dictionary.
    
    Args:
        model_config: Dictionary containing model configuration
            - provider: LLM provider name (ollama, openai, openrouter, huggingface)
            - api_key: API key for the provider
            - model_name: Name of the model to use
            - base_url: Base URL for the provider API (optional)
            - temperature: Temperature for text generation (default: 0.2)
            - max_tokens: Maximum tokens to generate (optional)
            
    Returns:
        Configured LLM service
    """
    if not model_config:
        return LLMService()
    
    # Extract provider from model config
    provider_name = model_config.get("provider", "ollama").lower()
    
    # Map provider name to enum
    provider_map = {
        "ollama": LLMProvider.OLLAMA,
        "openai": LLMProvider.OPENAI,
        "openrouter": LLMProvider.OPENROUTER,
        "huggingface": LLMProvider.HUGGINGFACE
    }
    
    provider = provider_map.get(provider_name, LLMProvider.OLLAMA)
    
    # Get API key
    api_key = model_config.get("api_key")
    
    # For OpenAI and OpenRouter, set the environment variable for CrewAI
    if provider in [LLMProvider.OPENAI, LLMProvider.OPENROUTER] and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Also set provider-specific environment variable
        if provider == LLMProvider.OPENROUTER:
            os.environ["OPENROUTER_API_KEY"] = api_key
    
    # Create config
    config = LLMConfig(
        provider=provider,
        api_key=api_key,
        model_name=model_config.get("model_name", os.getenv("OLLAMA_MODEL", "phi3")),
        base_url=model_config.get("base_url"),
        temperature=model_config.get("temperature", 0.2),
        max_tokens=model_config.get("max_tokens"),
        additional_params=model_config.get("additional_params", {})
    )
    
    return LLMService(config)