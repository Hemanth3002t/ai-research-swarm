import os

from dotenv import load_dotenv
from groq import Groq


class ResearchAgent:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY was not found.")

        self.client = Groq(api_key=api_key)

        self.model = "openai/gpt-oss-20b"

        self.system_prompt = """
You are a research assistant.

Your job is to:
1. Understand the user's research question.
2. Think about what information would be useful.
3. Provide a clear and structured answer.
4. Clearly distinguish facts from assumptions.
5. Do not invent sources or statistics.
"""

    def run(self, task):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": task,
                },
            ],
        )

        return response.choices[0].message.content