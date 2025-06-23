"""The configuration types for different LLM API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, root_validator

GPT3_MODEL_NAME = "text-davinci-003"
GPT4_MODEL_NAME = "gpt-4"
GPT4O_MODEL_NAME = "gpt-4o"

_LLM_PROXY_DEPLOYMENTS_AND_MODEL_VERSIONS = {
    "gpt-4o-20240806": "gpt-4o-2024-08-06",
    "gpt-4o-20240513": "gpt-4o-2024-05-13",
    "gpt-4": "gpt-4-0613",
    "gpt-4-turbo": "gpt-4-1106-preview",
    "gpt-35-turbo": "gpt-3.5-turbo-1106",
}

_LLM_PROXY_DEPLOYMENTS_AND_MODELS = {
    "gpt-4o-20240806": "gpt-4o",
    "gpt-4o-20240513": "gpt-4o",
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4",
    "gpt-35-turbo": "gpt-35-turbo",
}

_LLM_PROXY_DEFAULT_DEPLOYMENTS = {
    "gpt-4o": "gpt-4o-20240806",
    "gpt-4": "gpt-4-turbo",
    "gpt-35-turbo": "gpt-35-turbo",
}


class ModelConfig(BaseModel):
    """A wrapper around LLM model configurations."""

    temperature: float = 0.0
    max_tokens: int = 1000
    model: str
    timeout: int = 60
    max_retries: int = 3

    class Config:
        """Pydantic model configuration."""

        extra = "allow"



class AzureChatOpenAIConfig(ModelConfig):
    """A wrapper around langchain Azure OpenAI chat models' init parameters."""

    model: str
    azure_endpoint: str
    openai_api_version: str
    deployment_name: str
    openai_api_type: Literal["azure"] = "azure"

    model_version: Optional[str] = Field(None, exclude=True)


class PerflabLLMProxyConfig(AzureChatOpenAIConfig):
    """A wrapper around Perflab LLM proxy init parameters."""

    azure_endpoint: Literal["https://llm-proxy.perflab.nvidia.com"] = "https://llm-proxy.perflab.nvidia.com"
    openai_api_version: Literal["2024-05-01-preview"] = "2024-05-01-preview"

    model: Literal["gpt-4o", "gpt-4", "gpt-35-turbo"] = "gpt-4o"
    deployment_name: Literal["gpt-4o-20240806", "gpt-4o-20240513", "gpt-4", "gpt-4-turbo", "gpt-35-turbo"] = (
        "gpt-4o-20240806"
    )

    model_version: Literal[
        "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "gpt-4-0613", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"
    ] = Field("gpt-4o-2024-08-06", exclude=True)

    @root_validator(pre=True)  # pre=True to allow the validator to be called before field validation
    def set_deployment_model(cls, values):  # pylint: disable=no-self-argument # noqa: N805
        """Set deployment or model automatically."""

        model = values.get("model")
        deployment = values.get("deployment_name")
        model_version = values.get("model_version")
        if model and deployment:
            if _LLM_PROXY_DEPLOYMENTS_AND_MODELS[deployment] != model:
                raise ValueError(f'Incompatible model "{model}" and deployment "{deployment}" for llm-proxy')

            if model_version:
                if _LLM_PROXY_DEPLOYMENTS_AND_MODEL_VERSIONS[deployment] != model_version:
                    raise ValueError(
                        f'Incompatible model version "{model_version}" and deployment "{deployment}" for llm-proxy'
                    )
                return values

        if model and not deployment:
            values["deployment_name"] = deployment = _LLM_PROXY_DEFAULT_DEPLOYMENTS.get(model)

        elif deployment and not model:
            values["model"] = model = _LLM_PROXY_DEPLOYMENTS_AND_MODELS.get(deployment)

        if deployment and not model_version:
            values["model_version"] = _LLM_PROXY_DEPLOYMENTS_AND_MODEL_VERSIONS.get(deployment)

        return values 
    