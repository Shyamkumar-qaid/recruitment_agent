"""
API Key Manager Utility

This module provides functions for managing API keys across different LLM providers.
It handles setting environment variables and managing API key context.
"""

import os
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

def set_api_keys_from_config(model_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Set API keys from model configuration.
    
    Args:
        model_config: Dictionary containing model configuration including API keys
    """
    if not model_config:
        logger.debug("No model config provided, skipping API key setup")
        return
    
    # Extract provider and API key
    provider = model_config.get("provider", "ollama").lower()
    api_key = model_config.get("api_key")
    
    if not api_key:
        logger.debug(f"No API key provided for {provider}, skipping")
        return
    
    # Set API keys based on provider
    if provider in ["openai", "openrouter"]:
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info(f"Set OPENAI_API_KEY environment variable for {provider}")
        
        # Also set provider-specific environment variable
        if provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = api_key
            logger.info("Set OPENROUTER_API_KEY environment variable")
    
    elif provider == "huggingface":
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        logger.info("Set HUGGINGFACEHUB_API_TOKEN environment variable")