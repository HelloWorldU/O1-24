import os
from openai import OpenAI
from typing import Dict, List, Optional
from dotenv import load_dotenv
backends = {
    'qwen-math-plus-latest': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'moonshot-v1-8k': 'https://api.moonshot.cn/v1',
    'deepseek-chat': 'https://api.deepseek.com/v1',
}
load_dotenv()

class Evaluator:
    def __init__(self, backend='qwen-math-plus-latest', temperature=0):
        self.api_key = os.getenv(backend)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=backends[backend]
        )
        self.model = backend
        self.temperature = temperature
        
    def generate_response(self, input_text: str, backend: str=None) -> Dict[str, str]:
        try:
            if backend is None:
                client = self.client
                model = self.model
            else:
                client = OpenAI(
                    api_key=os.getenv(backend),
                    base_url=backends[backend]
                )
                model = backend
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text},
                ],
                temperature=self.temperature,
                stream=False
            )

            content = response.choices[0].message.content
            return content

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None
        

completion_tokens = prompt_tokens = 0

def deepseek_usage(backend='deepseek-chat'):
    global completion_tokens, prompt_tokens
    if backend == "deepseek-chat":
        cost = completion_tokens / 1000000 * 0.28 + prompt_tokens / 1000000 * 0.0014
    elif backend == "deepseek-reasoner":
        cost = completion_tokens / 1000000 * 2.19 + prompt_tokens / 1000000 * 0.14
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
