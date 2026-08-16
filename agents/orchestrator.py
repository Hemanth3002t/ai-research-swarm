import asyncio

from agents.agent import ResearchAgent
from database.repository import ResearchRepository


class ResearchSwarm:

    def __init__(self):

        # Database repository
        self.repository = ResearchRepository()

        # Primary researcher
        self.researcher = ResearchAgent(
            model="openai/gpt-oss-20b",
            max_completion_tokens=400,
        )

        # Fast specialist workers
        self.specialist = ResearchAgent(
            model="llama-3.1-8b-instant",
            max_completion_tokens=350,
        )

        # Writer
        self.writer = ResearchAgent(
            model="llama-3.1-8b-instant",
            max_completion_tokens=500,
        )

        # Critic
        self.critic = ResearchAgent(
            model="llama-3.1-8b-instant",
            max_completion_tokens=350,
        )

    # =========================================================
    # PARALLEL SPECIALIST AGENTS
    # =========================================================

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

        # Run specialist agents concurrently
        results = await asyncio.gather(
            *(run_agent(task) for task in tasks)
        )

        return results

    # =========================================================
    # CREATE DRAFT
    # =========================================================

    def create_draft(self, research, specialist_results):

        combined = "\n\n".join(
            f"--- Specialist {i + 1} ---\n{result}"
            for i, result in enumerate(specialist_results)
        )

        draft = self.writer.run(
            f"""
You are the lead technical writer.

Create a research report using ONLY the information below.

PRIMARY RESEARCH:

{research[:5000]}

SPECIALIST ANALYSIS:

{combined}

Requirements:

1. Use clear headings.
2. Organize information logically.
3. Separate confirmed facts from uncertain claims.
4. Do not invent facts.
5. Do not add information that is not supported by the research.
6. Keep the report concise.
""",
            use_web=False,
        )

        return draft

    # =========================================================
    # REVIEW DRAFT
    # =========================================================

    def review_draft(self, research, draft):

        review = self.critic.run(
            f"""
You are the Chief Research Quality Reviewer.

Your job is to strictly review the draft against the original
source research.

SOURCE RESEARCH:

{research[:5000]}

DRAFT REPORT:

{draft}

Check the draft for:

1. Factual contradictions.
2. Unsupported claims.
3. Invented statistics.
4. Misleading conclusions.
5. Missing important information.
6. Poor organization.
7. Claims that require stronger evidence.
8. Information that cannot be supported by the source research.

IMPORTANT:

Return ONLY a JSON object.

Do not use markdown.

Do not use ```json.

Do not include explanations before or after the JSON.

The JSON MUST contain exactly these three fields:

{{
    "status": "PASS",
    "critical_flaws": [],
    "revision_notes": ""
}}

OR:

{{
    "status": "FAIL",
    "critical_flaws": [
        "specific problem 1",
        "specific problem 2"
    ],
    "revision_notes": "Clear instructions explaining exactly what should be changed."
}}

Use PASS only when the report contains no significant unsupported
claims, factual contradictions, or misleading conclusions.

Use FAIL when there is any significant factual or evidence problem.

Be extremely strict.
""",
            use_web=False,
            json_mode=True,
        )

        return review

    # =========================================================
    # REVISE DRAFT
    # =========================================================

    def revise_draft(self, research, draft, review):

        if isinstance(review, dict):

            review_text = (
                f"Status: {review.get('status', 'FAIL')}\n"
                f"Critical flaws: "
                f"{review.get('critical_flaws', [])}\n"
                f"Revision notes: "
                f"{review.get('revision_notes', '')}"
            )

        else:

            review_text = str(review)

        revised = self.writer.run(
            f"""
You are revising a research report.

SOURCE RESEARCH:

{research[:5000]}

CURRENT DRAFT:

{draft}

QUALITY REVIEW:

{review_text}

Rewrite the report to fix every issue identified by the reviewer.

Rules:

1. Use only supported information.
2. Remove unsupported claims.
3. Clearly qualify uncertain information.
4. Do not invent replacement facts.
5. Keep the report concise and professional.
6. Preserve useful information from the original draft.
7. Do not add new claims that are not supported by the source research.
""",
            use_web=False,
        )

        return revised

    # =========================================================
    # MAIN SWARM
    # =========================================================

    async def run(self, topic):

        # -----------------------------------------------------
        # CREATE PROJECT
        # -----------------------------------------------------

        project_id = self.repository.create_project(topic)

        print(
            f"\nResearch project created: {project_id}"
        )

        # -----------------------------------------------------
        # STEP 1: PRIMARY RESEARCH
        # -----------------------------------------------------

        print(
            "\n[1/5] Primary researcher working..."
        )

        primary_task_id = self.repository.create_task(
            project_id=project_id,
            task_type="primary_research",
            description=topic,
            assigned_agent="PrimaryResearcher",
        )

        self.repository.update_task(
            primary_task_id,
            "running",
        )

        # Record primary researcher execution
        research_run_id = self.repository.create_agent_run(
            project_id=project_id,
            agent_name="PrimaryResearcher",
            task=topic,
            model="openai/gpt-oss-20b",
        )

        try:

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

Keep the research concise and structured.
""",
                use_web=True,
            )

            # Save successful agent run
            self.repository.complete_agent_run(
                research_run_id,
                research,
            )

            # Mark research task completed
            self.repository.update_task(
                primary_task_id,
                "completed",
                research,
            )

        except Exception as error:

            # Save failed agent run
            self.repository.fail_agent_run(
                research_run_id,
                error,
            )

            # Mark task failed
            self.repository.update_task(
                primary_task_id,
                "failed",
                str(error),
            )

            # Mark project failed
            self.repository.update_project_status(
                project_id,
                "failed",
            )

            raise

        # -----------------------------------------------------
        # STEP 2: PARALLEL SPECIALIST AGENTS
        # -----------------------------------------------------

        print(
            "[2/5] Parallel specialist agents working..."
        )

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
Analyze this research as a skeptical fact-checker.

Identify:

- claims requiring verification
- possible inconsistencies
- weak evidence
- unsupported conclusions
""",
        ]

        specialist_results = await self.run_parallel_agents(
            research,
            tasks,
        )

        # -----------------------------------------------------
        # STEP 3: CREATE INITIAL DRAFT
        # -----------------------------------------------------

        print(
            "[3/5] Writer creating draft..."
        )

        draft = self.create_draft(
            research,
            specialist_results,
        )

        # Save initial draft
        draft_id = self.repository.create_draft(
            project_id=project_id,
            content=draft,
            version=1,
        )

        # -----------------------------------------------------
        # STEP 4: QUALITY GATE
        # -----------------------------------------------------

        max_revisions = 2

        for attempt in range(max_revisions + 1):

            print(
                f"[4/5] Quality review "
                f"(attempt {attempt + 1})..."
            )

            review = self.review_draft(
                research,
                draft,
            )

            # Safely extract status
            if isinstance(review, dict):

                status = review.get(
                    "status",
                    "FAIL",
                )

            else:

                status = "FAIL"

            status = str(
                status
            ).upper().strip()

            print(
                f"    Review result: {status}"
            )

            # -------------------------------------------------
            # SAVE REVIEW
            # -------------------------------------------------

            if isinstance(review, dict):

                critical_flaws = review.get(
                    "critical_flaws",
                    [],
                )

                revision_notes = review.get(
                    "revision_notes",
                    "",
                )

            else:

                critical_flaws = [
                    "Reviewer returned invalid data."
                ]

                revision_notes = str(review)

            self.repository.create_review(
                project_id=project_id,
                draft_id=draft_id,
                status=status,
                critical_flaws=str(
                    critical_flaws
                ),
                revision_notes=str(
                    revision_notes
                ),
            )

            # -------------------------------------------------
            # PASS
            # -------------------------------------------------

            if status == "PASS":

                self.repository.update_project_status(
                    project_id,
                    "completed",
                )

                print(
                    "[5/5] Quality gate passed."
                )

                return draft

            # -------------------------------------------------
            # MAXIMUM REVISIONS
            # -------------------------------------------------

            if attempt == max_revisions:

                self.repository.update_project_status(
                    project_id,
                    "completed_with_warnings",
                )

                print(
                    "[5/5] Maximum revisions reached."
                )

                return draft

            # -------------------------------------------------
            # REVISION
            # -------------------------------------------------

            print(
                f"    Revising draft "
                f"(revision {attempt + 1})..."
            )

            draft = self.revise_draft(
                research,
                draft,
                review,
            )

            # Save revised draft
            draft_id = self.repository.create_draft(
                project_id=project_id,
                content=draft,
                version=attempt + 2,
            )

        return draft