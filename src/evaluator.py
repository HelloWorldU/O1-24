import os
import json
from openai import OpenAI
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class Evaluator:
    def __init__(self, backend='deepseek-chat', temperature=0.7):
        self.api_key = os.getenv("API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = backend
        self.temperature = temperature

    def generate_response(self, input_text: str) -> Dict[str, str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text},
                ],
                temperature=self.temperature,
                stream=False
            )

            if "reasoner" not in self.model:
                content = response.choices[0].message.content
                return content

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None
        

completion_tokens = prompt_tokens = 0

def deepseek_usage(backend='deepseek-chat'):
    global completion_tokens, prompt_tokens
    if backend == "deepseek-chat":
        cost = completion_tokens / 1000000 * 0.28 + prompt_tokens / 1000000 * 0.0014    # completed before eighth February, the discounted prices.
    elif backend == "deepseek-reasoner":
        cost = completion_tokens / 1000000 * 2.19 + prompt_tokens / 1000000 * 0.14
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
