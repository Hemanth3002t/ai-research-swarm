import os

from dotenv import load_dotenv
from groq import Groq


class ResearchAgent:
    def __init__(
        self,
        model="openai/gpt-oss-20b",
        max_completion_tokens=500,
    ):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY was not found.")

        self.client = Groq(api_key=api_key)

        self.model = model
        self.max_completion_tokens = max_completion_tokens

        self.system_prompt = """
You are a research assistant.

Your job is to:
1. Understand the research task.
2. Gather accurate and relevant information.
3. Use web search when requested.
4. Clearly separate facts from assumptions.
5. Do not invent sources, statistics, or facts.
6. Keep your output concise and structured.
"""

    def run(self, task, use_web=False):
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": task,
                },
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }

        # GPT-OSS supports reasoning_effort.
        # Llama models do not support this parameter.
        if self.model.startswith("openai/gpt-oss"):
            request["reasoning_effort"] = "low"

        # Enable browser search only when requested.
        if use_web:
            request["tool_choice"] = "required"
            request["tools"] = [
                {
                    "type": "browser_search"
                }
            ]

        response = self.client.chat.completions.create(**request)

        return response.choices[0].message.content