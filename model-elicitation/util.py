import ollama
from typing import Tuple, List, Dict
from rich import print
import os, re
from openai import OpenAI
from anthropic import Anthropic
import time

def inference(messages: List[Dict[str, str]] | str, client: ollama.Client|None=None,
              model: str='openai/gpt-oss-120b', temperature: float=0.0) -> Tuple[str | None, str]:
    """
    Perform inference using the specified Ollama model or GLaDoS (GTRI only).

    Parameters
    ----------
    messages : List[str] | str
        A list of message strings to send to the model as the conversation history.
        Alternatively, just one message, in which case it's passed as a user.
    client : ollama.Client | None
        An initialized Ollama client used to communicate with the model.
    model : str, optional
        The model identifier to use for inference (default is ``'gemma3:27b'``).
    temperature : float, optional
        Sampling temperature for the model; ``0.0`` produces deterministic output
        (default).

    Returns
    -------
    Tuple[str, str]
        A tuple containing ``thinking`` (the model's internal reasoning or
        explanation) and ``content`` (the generated response).

    Raises
    ------
    Exception
        Propagates any exception raised by the Ollama client or GLaDoS during the chat
        request, after printing a formatted error message.
    """
    if isinstance(messages, str):
        messages = [{"role": "user","content": messages}]
    try:
        if client is not None:
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
        elif model.startswith("claude"):
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            ant_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            max_retries = 10
            base_delay = 1
            for attempt in range(max_retries):
                try:
                    message = ant_client.messages.create(
                        max_tokens=2000,
                        messages=messages,
                        model=model,
                    )
                    return (None, message.content[0].text)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f'[yellow]Anthropic API error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...[/]')
                    time.sleep(delay)
        else:
            base_url = "https://glados.ctisl.gtri.org"
            if "LITELLM_API_KEY" not in os.environ:
                raise ValueError("LITELLM_API_KEY environment variable not set")
            api_key = os.environ["LITELLM_API_KEY"]
            openai_client = OpenAI(api_key=api_key, base_url=base_url)

            chat_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return (None, chat_response.choices[0].message.content)
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
