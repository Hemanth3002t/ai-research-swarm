import os
import json
import re

from dotenv import load_dotenv
from groq import Groq


class ResearchAgent:

    # =========================================================
    # CLEAN MODEL OUTPUT
    # =========================================================

    def clean_output(self, content):
        """
        Remove internal reasoning and formatting artifacts
        from model output before returning it to the application.
        """

        if content is None:
            return ""

        content = str(content).strip()

        if not content:
            return ""

        # -----------------------------------------------------
        # Remove <think>...</think>
        # -----------------------------------------------------

        pattern = re.compile(
            r"<think>.*?</think>",
            re.IGNORECASE | re.DOTALL,
        )

        content = pattern.sub("", content)

        # -----------------------------------------------------
        # Remove <analysis>...</analysis>
        # -----------------------------------------------------

        pattern = re.compile(
            r"<analysis>.*?</analysis>",
            re.IGNORECASE | re.DOTALL,
        )

        content = pattern.sub("", content)

        # -----------------------------------------------------
        # Handle unclosed <think>
        # -----------------------------------------------------

        lower_content = content.lower()

        if "<think>" in lower_content:

            start = lower_content.find("<think>")

            content = content[:start]

        # -----------------------------------------------------
        # Handle unclosed <analysis>
        # -----------------------------------------------------

        lower_content = content.lower()

        if "<analysis>" in lower_content:

            start = lower_content.find("<analysis>")

            content = content[:start]

        content = content.strip()

        # -----------------------------------------------------
        # Remove markdown code fences
        # -----------------------------------------------------

        if (
            content.startswith("```")
            and content.endswith("```")
        ):

            lines = content.splitlines()

            if len(lines) >= 3:

                content = "\n".join(
                    lines[1:-1]
                ).strip()

        return content.strip()

    # =========================================================
    # CLEAN JSON RESPONSE
    # =========================================================

    def clean_json_response(self, content):
        """
        Clean an AI JSON response before parsing it.

        This handles accidental markdown fences and
        whitespace without modifying the JSON structure.
        """

        if content is None:
            return ""

        content = str(content).strip()

        if not content:
            return ""

        # Remove markdown fences such as:
        #
        # ```json
        # {...}
        # ```

        if content.startswith("```"):

            lines = content.splitlines()

            if len(lines) >= 3:

                content = "\n".join(
                    lines[1:-1]
                ).strip()

        return content

    # =========================================================
    # INITIALIZE AGENT
    # =========================================================

    def __init__(
        self,
        model="openai/gpt-oss-20b",
        max_completion_tokens=500,
    ):

        # =====================================================
        # LOAD ENVIRONMENT
        # =====================================================

        load_dotenv()

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY was not found."
            )

        # =====================================================
        # GROQ CLIENT
        # =====================================================

        self.client = Groq(
            api_key=api_key
        )

        self.model = model

        self.max_completion_tokens = (
            max_completion_tokens
        )

        # =====================================================
        # SYSTEM PROMPT
        # =====================================================

        self.system_prompt = """
You are a research assistant.

Your job is to:

1. Understand the research task.
2. Gather accurate and relevant information.
3. Use web search when requested.
4. Clearly separate facts from assumptions.
5. Do not invent sources, statistics, URLs, or facts.
6. Keep your output concise and structured.
7. When sources are requested, provide only sources
   actually identified during the research process.
8. Never expose internal reasoning.
9. Do not output <think> or <analysis> blocks.
"""

    # =========================================================
    # RUN AGENT
    # =========================================================

    def run(
        self,
        task,
        use_web=False,
        json_mode=False,
    ):

        # =====================================================
        # BUILD REQUEST
        # =====================================================

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

            "max_completion_tokens":
                self.max_completion_tokens,
        }

        # =====================================================
        # REASONING
        # =====================================================

        # GPT-OSS models support reasoning_effort.

        if self.model.startswith(
            "openai/gpt-oss"
        ):

            request["reasoning_effort"] = "low"

        # =====================================================
        # JSON MODE
        # =====================================================

        if json_mode:

            request["response_format"] = {
                "type": "json_object"
            }

        # =====================================================
        # WEB SEARCH
        # =====================================================

        # IMPORTANT:
        #
        # Groq does not allow JSON mode + tool/function
        # calling together.
        #
        # Therefore web search is only enabled when
        # json_mode is False.

        if use_web and not json_mode:

            request["tool_choice"] = "required"

            request["tools"] = [
                {
                    "type": "browser_search"
                }
            ]

        # =====================================================
        # API REQUEST
        # =====================================================

        try:

            response = (
                self.client
                .chat
                .completions
                .create(**request)
            )

        except Exception:

            # Re-raise the original Groq error so that
            # the orchestrator can handle/log it.

            raise

        # =====================================================
        # VALIDATE RESPONSE
        # =====================================================

        if not response.choices:

            raise RuntimeError(
                f"{self.model} returned no choices."
            )

        message = response.choices[0].message

        if message is None:

            raise RuntimeError(
                f"{self.model} returned no message."
            )

        content = message.content

        # =====================================================
        # EMPTY RESPONSE CHECK
        # =====================================================

        if content is None:

            raise RuntimeError(
                f"{self.model} returned an empty response."
            )

        content = str(content).strip()

        if not content:

            raise RuntimeError(
                f"{self.model} returned an empty response."
            )

        # =====================================================
        # JSON RESPONSE
        # =====================================================

        if json_mode:

            json_content = (
                self.clean_json_response(
                    content
                )
            )

            if not json_content:

                return {
                    "status": "FAIL",
                    "critical_flaws": [
                        "The AI returned an empty JSON response."
                    ],
                    "revision_notes": (
                        "Generate a valid JSON object "
                        "using the required schema."
                    ),
                }

            try:

                parsed = json.loads(
                    json_content
                )

                # -------------------------------------------------
                # JSON must be an object
                # -------------------------------------------------

                if isinstance(
                    parsed,
                    dict,
                ):

                    return parsed

                return {
                    "status": "FAIL",
                    "critical_flaws": [
                        (
                            "The AI returned valid JSON, "
                            "but it was not a JSON object."
                        )
                    ],
                    "revision_notes": (
                        "Return a JSON object with "
                        "the required fields."
                    ),
                }

            except (
                json.JSONDecodeError,
                TypeError,
            ):

                return {
                    "status": "FAIL",
                    "critical_flaws": [
                        "The AI returned invalid JSON."
                    ],
                    "revision_notes": [
                        "Return ONLY valid JSON. "
                        "Do not include markdown or explanations."
                    ],
                    "raw_response": json_content[:2000],
                }

        # =====================================================
        # NORMAL TEXT RESPONSE
        # =====================================================

        cleaned_content = self.clean_output(
            content
        )

        # =====================================================
        # FINAL EMPTY CHECK AFTER CLEANING
        # =====================================================

        if not cleaned_content:

            raise RuntimeError(
                f"{self.model} returned only internal "
                "reasoning or formatting with no usable "
                "text response."
            )

        return cleaned_content