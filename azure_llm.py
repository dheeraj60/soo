import os
import json
import time
import logging
from dotenv import load_dotenv
from typing import Dict, Any
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

KEY_VAULT_URL = "https://fstodevazureopenai.vault.azure.net/"

def _load_config_from_vault() -> Dict[str, Any]:
    """Load API key and endpoint from Azure Key Vault with retry logic"""
    kv_url = KEY_VAULT_URL
    api_key = None
    endpoint = None
    api_version = None
    model = None

    max_retries = 3
    retry_delay = 0.5

    for attempt in range(max_retries):
        try:
            credential = DefaultAzureCredential()
            kvclient = SecretClient(vault_url=kv_url, credential=credential)

            try:
                api_key = kvclient.get_secret("llm-api-key").value
                logger.info(f"API Key loaded from Key Vault (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"Failed to load API key from Key Vault (attempt {attempt + 1}/{max_retries}): {e}")
                raise ValueError(f"Failed to load API key from Key Vault: {e}")

            try:
                endpoint_secret = kvclient.get_secret("llm-base-endpoint")
                endpoint = endpoint_secret.value
                api_version_secret = kvclient.get_secret("llm-mini-version")
                api_version = api_version_secret.value
                model_secret = kvclient.get_secret("llm-41")
                model = model_secret.value
                logger.info("Endpoint loaded from Key Vault")
            except Exception as e:
                logger.warning(f"Failed to load endpoint from Key Vault: {e}; using default endpoint")
                endpoint = "https://stg-secureapi.hexaware.com/api/azureai"

            break

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to load Azure config (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Failed to load Azure config after {max_retries} attempts: {e}")
                raise ValueError(f"Failed to load API key from Key Vault after {max_retries} attempts: {e}")

    config = {
        "api_key": api_key,
        "endpoint": endpoint,
        "api_version": api_version,
        "model": model,
    }

    if config.get("api_key"):
        logger.info(f"API Key loaded, starts with: {config['api_key'][:5]}...")
        logger.info(f"Azure OpenAI config - Model: {config['model']}, Endpoint: {config['endpoint']}")
    else:
        logger.error("Failed to load API key from Key Vault")

    return config

def build_llm_config(temperature: float = 0.3) -> Dict[str, Any]:
    """Build Azure OpenAI config using credentials from Azure Key Vault"""
    config = _load_config_from_vault()

    api_key = config.get("api_key")
    endpoint = config.get("endpoint")
    api_version = config.get("api_version")
    model = config.get("model")

    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY from Key Vault")
    if not endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT from Key Vault")

    print(f"[LLM CONFIG] Azure OpenAI | Model: {model} | Endpoint: {endpoint}")

    # Return all necessary config to use with your OpenAI/AzureOpenAI client
    return {
        "model": model,
        "api_key": api_key,
        "endpoint": endpoint,
        "api_version": api_version,
        "temperature": temperature,
        "timeout": 180,
    }

llm_config = build_llm_config()


def evaluate_with_azure_llm(
        prompt: str,
        cache_path: str,
        max_retries: int = 3,
        backoff_factor: int = 2,
        request_timeout: int = 30,
        logger=None
) -> dict:
    _log = logger or logging

    # Load Azure OpenAI credentials and config from Key Vault
    llm_config = build_llm_config(temperature=0)  # Or set temperature as needed

    model = llm_config['model']
    api_key = llm_config['api_key']
    endpoint = llm_config['endpoint']
    api_version = llm_config['api_version']

    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{endpoint}/openai/deployments/{model}"
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=llm_config.get('temperature', 0),
                timeout=request_timeout,
                messages=[
                    {"role": "system", "content": (
                        "You are an autonomous debugging agent. Your goal is to analyze problems, "
                        "fix code, and ensure tests pass. Always return ONLY valid JSON with no explanations, "
                        "no markdown, and no extra text."
                    )},
                    {"role": "user", "content": prompt},
                ],
            )

            raw = response.choices[0].message.content.strip()
            print("\n[DEBUG] Raw LLM response:\n", raw, "\n")  # <--- Debug print

            # Robustly strip markdown and whitespace
            if raw.startswith("```json"):
                raw = raw[len("```json"):].strip()
            if raw.startswith("```"):
                raw = raw[3:].strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()

            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(raw)

            # Try parsing JSON
            try:
                return json.loads(raw)
            except json.JSONDecodeError as jde:
                _log.warning("JSON decode error: %s\nRaw content: %r", jde, raw)
                # Optionally, save the raw response to a .txt file for manual inspection
                with open(cache_path + ".txt", "w", encoding="utf-8") as ftxt:
                    ftxt.write(raw)
                raise

        except Exception as e:
            wait = (backoff_factor ** attempt) + (0.25 * (attempt + 1))
            _log.warning("Azure LLM error: %s (attempt %d/%d). Retrying in %.2fs...",
                         str(e), attempt + 1, max_retries, wait)
            time.sleep(wait)

    _log.error("All retries exhausted for Azure LLM.")
    return {"answer": "[API Error: all retries exhausted]"}
