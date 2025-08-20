import asyncio
from abc import ABC, abstractmethod
import os, time
from typing import Optional, Tuple

from textarena.core import Agent
import textarena as ta 

__all__ = ["OpenRouterAgent", "OpenAIAgent", "HumanAgent"]
STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."
    

class OpenRouterAgent(Agent):
    """ Agent class using the OpenRouter API to generate responses. """
    def __init__(self, model_name: str, system_prompt: Optional[str] = STANDARD_GAME_PROMPT, verbose: bool = False, **kwargs):
        """
        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT)
            verbose (bool): If True, additional debug info will be printed.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        super().__init__()
        self.model_name = model_name 
        self.verbose = verbose 
        self.system_prompt = system_prompt
        self.kwargs = kwargs
        self.think = ""
        try:
            from openai import OpenAI
            from openai._exceptions import OpenAIError
        except ImportError:
            raise ImportError("OpenAI package is required for OpenRouterAgent. Install it with: pip install openai")
        
        api_key = os.getenv("OPENROUTER_API_KEY") # Set the open router api key from an environment variable
        if not api_key:
            raise ValueError("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def _make_request(self, observation: str) -> str:
        """ Make a single API request to OpenRouter and return the generated message. """
        # messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": observation}]
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + observation}]
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, n=1, stop=None, **self.kwargs)
        return response.choices[0].message.content.strip()

    def _retry_request(self, observation: str, retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(self, observation: str) -> str:
        """
        Process the observation using the OpenRouter API and return the action.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return self._retry_request(observation)
        
    def act_full(self, observation: str) -> Tuple[str, str, str, dict, float]:
        """
        Process the observation and return the full action information.
        This method is compatible with the patched_run_game_impl function.
        
        Args:
            observation (str): The input string to process.
            
        Returns:
            Tuple[str, str, str, dict, float]: A tuple containing:
                - raw: The raw response from the model
                - extracted: The extracted action (same as raw for OpenRouter agents)
                - prompt: The prompt sent to the model
                - format_feedback: Empty dict as format feedback
                - logp: Log probability (0.0 as placeholder)
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
            
        # Get the raw response
        raw = self._retry_request(observation)
        
        # For OpenRouter agents, extracted action is the same as raw response
        extracted = raw
        
        # The prompt is the observation prepended with system prompt
        prompt = f"{self.system_prompt}\n\n{observation}"
        
        # No format feedback for OpenRouter agents
        format_feedback = {}
        
        # No log probability for OpenRouter agents
        logp = 0.0
        
        # print("act_full❀", self.model_name, "✈", observation[:50], "🎈", raw[:50])
        
        return raw, extracted, prompt, format_feedback, logp


class OpenAIAgent(Agent):
    """Agent class using the OpenAI API to generate responses."""

    def __init__(self, model_name: str, system_prompt: Optional[str]=STANDARD_GAME_PROMPT, verbose: bool=False, api_key: str|None=None, base_url: str|None=None,**kwargs):
        """
        Initialize the OpenAI agent.
        
        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT).
            verbose (bool): If True, additional debug info will be printed.
            api_key (str | None): The API key for the OpenAI API.
            base_url (str | None): The base URL for the OpenAI API.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.kwargs = kwargs
        self.think = ""
        return
        try: from openai import OpenAI, AzureOpenAI
        except ImportError: raise ImportError("OpenAI package is required for OpenAIAgent. Install it with: pip install openai")
        # if "gpt-5" in model_name.lower() or "gpt5" in model_name.lower():
        #     from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        #     endpoint = "https://haotian-east-us-2.openai.azure.com/"
        #     token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        #     api_version = "2024-12-01-preview"

        #     self.client = AzureOpenAI(
        #         api_version=api_version,
        #         azure_endpoint=endpoint,
        #         azure_ad_token_provider=token_provider,
        #     )
            

        if "gpt-5" in model_name.lower() or "gpt5" in model_name.lower() or "gpt-4o" in model_name.lower() or "o4-mini" in model_name.lower():
            
            from azure.identity import (
                ChainedTokenCredential,
                AzureCliCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider
            )
            if "gpt-5" in model_name.lower() or "o4-mini" in model_name.lower():
                api_version = '2024-12-01-preview'
            else:
                api_version = '2024-10-21'
            def deploy_model(model_name: str) -> str:
                if "gpt-5" == model_name:
                    return "gpt-5_2025-08-07"
                elif "gpt-5-chat" == model_name:
                    return "gpt-5-chat_2025-08-07"
                elif "gpt-5-mini" == model_name:
                    return "gpt-5-mini_2025-08-07"
                elif "gpt-5-nano" == model_name:
                    return "gpt-5-nano_2025-08-07"
                elif "gpt-4o" == model_name:
                    return "gpt-4o_2024-11-20"
                elif "gpt-4o-mini" == model_name:
                    return "gpt-4o-mini_2024-07-18"
                else:
                    return "gpt-5-chat_2025-08-07"
            self.model_name = deploy_model(model_name)
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            scope = "api://trapi/.default"
            credential = get_bearer_token_provider(
                ChainedTokenCredential(
                    AzureCliCredential(),
                    ManagedIdentityCredential(),
                ),
                scope
            )  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                azure_endpoint="https://trapi.research.microsoft.com/gcr/shared",
                azure_ad_token_provider=credential,
                api_version=api_version
            )
        else:
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY", "123")
            if base_url is None:
                base_url = os.getenv("OPENAI_BASE_URL", "") # Default to empty string if not set
            if "32b" in model_name.lower():
                self.client = OpenAI(api_key="123", base_url="http://0.0.0.0:9998/v1")
            elif "qwen3-8b-v1-lora-0812-3epochs" in model_name.lower():
                self.client = OpenAI(api_key="123", base_url="http://0.0.0.0:9996/v1")
            elif "8b" in model_name.lower():
                self.client = OpenAI(api_key="123", base_url="http://0.0.0.0:9999/v1")
            else:
                self.client = OpenAI(api_key="123", base_url="http://0.0.0.0:9998/v1")
            # if "azure" in base_url.lower():
            #     self.client = AzureOpenAI(api_key=api_key, base_url=base_url, api_version="2025-01-01-preview")
            # else:
            #     self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def _make_request(self, observation: str) -> str:
        """
        Make a single API request to OpenAI and return the generated message.
        
        Args:
            observation (str): The input string to process.
        
        Returns:
            str: The generated response text.
        """
        # messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": observation}]
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + observation}]
        
        # Make the API call using the provided model and messages.
        
        # completion = self.client.chat.completions.create(model=self.model_name, messages=messages, n=1, stop=None, **self.kwargs)
        response = ""
        # response = completion.choices[0].message.content.strip()
        # if "<think>" in response and "</think>" in response:
        #     self.think = response.split("<think>")[1].split("</think>")[0].strip()
        #     response = response.split("</think>")[-1].strip()
        # else:
        #     self.think = ""
        return response
    
    def _retry_request(self, observation: str, retries: int=3, delay: int=2) -> str:
        """
        Attempt to make an API request with retries.
        
        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.
        
        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        return "[API FAILED]" + str(last_exception)
    
    def __call__(self, observation: str) -> str:
        """
        Process the observation using the OpenAI API and return the generated response.
        
        Args:
            observation (str): The input string to process.
        
        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        result = self._retry_request(observation)
        # print(self.model_name, "✈", observation[:50], "🎈", result[:50])
        return result
        
    def act_full(self, observation: str) -> Tuple[str, str, str, dict, float]:
        """
        Process the observation and return the full action information.
        This method is compatible with the patched_run_game_impl function.
        
        Args:
            observation (str): The input string to process.
            
        Returns:
            Tuple[str, str, str, dict, float]: A tuple containing:
                - raw: The raw response from the model
                - extracted: The extracted action (same as raw for OpenAI agents)
                - prompt: The prompt sent to the model
                - format_feedback: Empty dict as format feedback
                - logp: Log probability (0.0 as placeholder)
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
            
        # Get the raw response
        raw = self._retry_request(observation)
        
        # For OpenAI agents, extracted action is the same as raw response
        extracted = raw
        
        # The prompt is the observation prepended with system prompt
        prompt = f"{self.system_prompt}\n\n{observation}"
        
        # No format feedback for OpenAI agents
        format_feedback = {}
        
        # No log probability for OpenAI agents
        logp = 0.0
        
        # print("act_full❀", self.model_name, "✈", observation[:50], "🎈", raw[:50])
        
        return raw, extracted, prompt, format_feedback, logp


class HumanAgent(Agent):
    """ Human agent class that allows the user to input actions manually """
    def __init__(self):
        super().__init__()

    def __call__(self, observation: str) -> str:
        """
        Process the observation and return the action.
        
        Args:
            observation (str): The input string to process.
            
        Returns:
            str: The response generated by the agent.
        """
        print("\n\n+++ +++ +++") # for easies visualization of what is part of each turns observation
        return input(f"Current observations: {observation}\nPlease enter the action: ")


# OpenAIAgent1 = OpenAIAgent(model_name="gpt-5-chat", system_prompt=STANDARD_GAME_PROMPT, verbose=True)

# a = OpenAIAgent1._make_request(observation="Hello, how are you?")
# print(a)