import os
import json

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

    def run(self, task, use_web=False, json_mode=False):

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
        if self.model.startswith("openai/gpt-oss"):
            request["reasoning_effort"] = "low"

        # Enable JSON response when requested.
        if json_mode:
            request["response_format"] = {
                "type": "json_object"
            }

        # Enable browser search when requested.
        if use_web:
            request["tool_choice"] = "required"
            request["tools"] = [
                {
                    "type": "browser_search"
                }
            ]

        response = self.client.chat.completions.create(**request)

        content = response.choices[0].message.content

        # Convert JSON string into a Python dictionary.
        if json_mode:

            try:
                parsed = json.loads(content)

                # Make sure the result is actually a dictionary.
                if isinstance(parsed, dict):
                    return parsed

                return {
                    "status": "FAIL",
                    "critical_flaws": [
                        "Critic returned JSON, but it was not an object."
                    ],
                    "revision_notes": (
                        "Return a JSON object with status, "
                        "critical_flaws and revision_notes."
                    ),
                }

            except (json.JSONDecodeError, TypeError):

                return {
                    "status": "FAIL",
                    "critical_flaws": [
                        "Critic returned invalid JSON."
                    ],
                    "revision_notes": (
                        "Return only a valid JSON object."
                    ),
                }

        return content