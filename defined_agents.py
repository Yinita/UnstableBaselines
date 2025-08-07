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

        try: from openai import OpenAI, AzureOpenAI
        except ImportError: raise ImportError("OpenAI package is required for OpenAIAgent. Install it with: pip install openai")

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
        if "azure" in base_url.lower():
            self.client = AzureOpenAI(api_key=api_key, base_url=base_url, api_version="2025-01-01-preview")
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
    
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
        completion = self.client.chat.completions.create(model=self.model_name, messages=messages, n=1, stop=None, **self.kwargs)
        response = completion.choices[0].message.content.strip()
        if "<think>" in response and "</think>" in response:
            self.think = response.split("<think>")[1].split("</think>")[0].strip()
            response = response.split("</think>")[-1].strip()
        else:
            self.think = ""
        return response
    
    def _retry_request(self, observation: str, retries: int=3, delay: int=5) -> str:
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
        Process the observation using the OpenAI API and return the generated response.
        
        Args:
            observation (str): The input string to process.
        
        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return self._retry_request(observation)


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
