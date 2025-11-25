import ollama
from typing import Tuple, List, Dict, Callable
from rich import print
import os, re
from openai import OpenAI
from anthropic import Anthropic
import google.genai as gemini
import time

def _retry_with_backoff(func: Callable, api_name: str, max_retries: int = 10, base_delay: int = 1):
    """Helper function to retry API calls with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f'[yellow]{api_name} API error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...[/]')
            print(f'[yellow]{e}[/]')
            time.sleep(delay)

def inference(messages: List[Dict[str, str]] | str, client: ollama.Client|None=None,
              model: str='openai/gpt-oss-120b', temperature: float=0.0) -> Tuple[str | None, str]:
    """
    Perform inference using various LLM providers including Ollama, Anthropic Claude,
    Google Gemini, OpenAI GPT, or LiteLLM.

    Parameters
    ----------
    messages : List[Dict[str, str]] | str
        A list of message dictionaries with 'role' and 'content' keys representing
        the conversation history. If a single string is provided, it's converted to
        a user message.
    client : ollama.Client | None
        An initialized Ollama client. If provided, uses Ollama for inference. If None,
        the function determines the provider based on the model name prefix.
    model : str, optional
        The model identifier to use for inference (default is ``'openai/gpt-oss-120b'``).
        Model prefixes determine the provider: 'claude-*' for Anthropic, 'gemini-*' for
        Google, 'gpt-*' for OpenAI, or other prefixes for LiteLLM.
    temperature : float, optional
        Sampling temperature for the model; ``0.0`` produces deterministic output
        (default). Only used with Ollama and LiteLLM providers.

    Returns
    -------
    Tuple[str | None, str]
        A tuple containing ``thinking`` (the model's internal reasoning, if available;
        otherwise None) and ``content`` (the generated response text).

    Raises
    ------
    ValueError
        If required API keys are not set in environment variables (ANTHROPIC_API_KEY,
        GEMINI_API_KEY, OPENAI_API_KEY, or LITELLM_API_KEY).
    Exception
        Propagates any exception raised by the API clients during inference, after
        printing a formatted error message and retrying with exponential backoff.
    
    Notes
    -----
    The function automatically retries failed API calls up to 10 times with exponential 
    backoff to handle transient errors.
    """
    if isinstance(messages, str):
        messages = [{"role": "user","content": messages}]
    
    try:
        if client is not None:
            def _ollama_call():
                response = client.chat(
                    model=model,
                    messages=messages,
                    options={'temperature': temperature}
                ).message
                if response.content is None:
                    raise ValueError("No content in response")
                if hasattr(response, 'thinking'):
                    return (response.thinking, response.content)
                else:
                    return (None, response.content)
            return _retry_with_backoff(_ollama_call, "Ollama")
        
        elif model.startswith("claude"):
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            ant_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            
            def _claude_call():
                message = ant_client.messages.create(
                    max_tokens=2000,
                    messages=messages,
                    model=model,
                )
                return (None, message.content[0].text)
            return _retry_with_backoff(_claude_call, "Anthropic")
        
        elif model.startswith("gemini"):
            if "GEMINI_API_KEY" not in os.environ:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            gdm_client = gemini.Client()
            
            def _gemini_call():
                response = gdm_client.models.generate_content(
                    model=model,
                    contents=messages[0]['content']
                )
                gdm_client.close()
                return (None, response.text)
            return _retry_with_backoff(_gemini_call, "Gemini")
        
        elif model.startswith("gpt"):
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

            def _openai_call():
                chat_response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return (None, chat_response.choices[0].message.content)
            return _retry_with_backoff(_openai_call, "OpenAI")
        
        else:
            base_url = "https://glados.ctisl.gtri.org"
            if "LITELLM_API_KEY" not in os.environ:
                raise ValueError("LITELLM_API_KEY environment variable not set")
            api_key = os.environ["LITELLM_API_KEY"]
            openai_client = OpenAI(api_key=api_key, base_url=base_url)

            def _litellm_call():
                chat_response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return (None, chat_response.choices[0].message.content)
            return _retry_with_backoff(_litellm_call, "LITELLM")
    
    except Exception as e:
        print(f'[red]util/inference :: messages: {messages}\nclient: {client is not None}\nmodel: {model}, temp: {temperature}[/]')
        raise e
    
def safe_name(name: str, n:int=5) -> str:
    """
    Create a safe filename by replacing non-alphanumeric characters with underscores.
    Parameters
    ---------- 
    name : str
        The original name string to be sanitized.
    n : int, optional
        The maximum length of the sanitized name (default is 5).
    Returns
    -------
    str
        The sanitized name suitable for use in filenames.
    """
    try:
        if n <= 0:
            raise ValueError("n must be positive")
        if name is None:
            raise ValueError("name must exist")
        return re.sub(r'[^a-zA-Z0-9]', '_', name[:min(n, len(name))])
    except Exception as e:
        print(f'[red]util/safe_name :: name: {name} :: n: {n}[/]')
        
def extract_str(filename: str) -> str:
    """
    Read the contents of a text or markdown file and return it as a string.

    Parameters
    ----------
    filename : str
        Path to the file to be read.

    Returns
    -------
    str
        The full contents of the file as a string.

    Raises
    ------
    Exception
        Any exception raised while opening or reading the file is caught and
        results in an empty string being returned. Also raised if not a .md
        or .txt file.
    """
    try:
        if not (filename.endswith('.md') or filename.endswith('.txt')):
            raise Exception('Not a .txt or .md file')
        with open(filename) as f:
            return f.read()
    except Exception as e:
        print(f'[red]ERROR util/extract_str :: filename: {filename}')
        raise e
