import asyncio

from agents.agent import ResearchAgent


class ResearchSwarm:
    def __init__(self):

        # Stronger model for primary web research
        self.researcher = ResearchAgent(
            model="openai/gpt-oss-20b",
            max_completion_tokens=400,
        )

        # Faster/lighter model for parallel specialist workers
        self.specialist = ResearchAgent(
            model="llama-3.1-8b-instant",
            max_completion_tokens=500,
        )

        # Lighter model for final synthesis
        self.editor = ResearchAgent(
            model="llama-3.1-8b-instant",
            max_completion_tokens=500,
        )

    async def run_parallel_agents(self, research, tasks):

        async def run_agent(task):

            prompt = f"""
Here is research collected by the primary researcher:

{research[:6000]}

Your assignment:

{task}

Analyze only the information provided above.

Do not invent facts.

Return concise findings in bullet points.
"""

            return await asyncio.to_thread(
                self.specialist.run,
                prompt,
                False,
            )

        results = await asyncio.gather(
            *(run_agent(task) for task in tasks)
        )

        return results

    async def run(self, topic):

        print("\n[1/3] Primary researcher working...")

        research = self.researcher.run(
            f"""
Research the following topic using current web sources:

{topic}

Collect:
- important facts
- recent developments
- important companies or tools
- relevant dates
- useful evidence

Keep the research concise.
""",
            use_web=True,
        )

        print("[2/3] Parallel specialist agents working...")

        tasks = [
            """
Analyze this research from a technical/software engineering
perspective.

Identify:
- technologies
- tools
- architectures
- technical developments
""",

            """
Analyze this research from a market and industry perspective.

Identify:
- companies
- products
- adoption trends
- industry impact
""",

            """
Act as a skeptical fact-checker.

Identify:
- claims that need verification
- possible inconsistencies
- weak evidence
- unsupported conclusions
""",
        ]

        specialist_results = await self.run_parallel_agents(
            research,
            tasks,
        )

        print("[3/3] Combining specialist findings...")

        combined = "\n\n".join(
            f"--- Specialist {i + 1} ---\n{result}"
            for i, result in enumerate(specialist_results)
        )

        final = self.editor.run(
            f"""
You are the lead research editor.

Original research:

{research[:5000]}

Specialist analysis:

{combined}

Create a concise final research report.

Requirements:
- Use only information contained in the research.
- Use the specialist findings to organize the report.
- Clearly identify uncertainty.
- Do not invent information.
- Use clear headings and bullet points.
""",
            use_web=False,
        )

        return final