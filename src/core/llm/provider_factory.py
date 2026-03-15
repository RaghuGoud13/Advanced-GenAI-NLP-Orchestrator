from enum import Enum
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel

class ProviderType(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"

class LLMProviderFactory:
    """
    Factory pattern implementation to support multiple LLM backends.
    """
    @staticmethod
    def create(
        provider: ProviderType, 
        config: Dict[str, Any]
    ) -> BaseChatModel:
        """
        Create a chat model instance based on the provider type.
        
        Args:
            provider (ProviderType): The type of LLM provider.
            config (Dict[str, Any]): Configuration parameters for the provider.
            
        Returns:
            BaseChatModel: An instance of a LangChain chat model.
        """
        if provider == ProviderType.AZURE_OPENAI:
            return AzureChatOpenAI(
                azure_endpoint=config["azure_endpoint"],
                azure_deployment=config["azure_deployment"],
                api_key=config["api_key"],
                api_version=config["api_version"],
                temperature=config.get("temperature", 0.0)
            )
        elif provider == ProviderType.OPENAI:
            return ChatOpenAI(
                api_key=config["api_key"],
                model=config.get("model_name", "gpt-4"),
                temperature=config.get("temperature", 0.0)
            )
        elif provider == ProviderType.ANTHROPIC:
            return ChatAnthropic(
                anthropic_api_key=config["api_key"],
                model_name=config.get("model_name", "claude-3-opus-20240229"),
                temperature=config.get("temperature", 0.0)
            )
        elif provider == ProviderType.BEDROCK:
            return ChatBedrock(
                model_id=config["model_id"],
                region_name=config.get("region_name", "us-east-1"),
                credentials_profile_name=config.get("credentials_profile_name"),
                model_kwargs=config.get("model_kwargs", {})
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
