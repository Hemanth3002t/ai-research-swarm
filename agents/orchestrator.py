import asyncio

from agents.agent import ResearchAgent
from database.repository import ResearchRepository


class ResearchSwarm:

    def __init__(self):

        # =====================================================
        # DATABASE
        # =====================================================

        self.repository = ResearchRepository()

        # =====================================================
        # AI AGENTS
        # =====================================================

        # Primary researcher
        self.researcher = ResearchAgent(
            model="openai/gpt-oss-20b",
            max_completion_tokens=400,
        )

        # Specialist workers
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

    async def run_parallel_agents(
        self,
        project_id,
        research,
        tasks,
    ):

        async def run_agent(agent_config):

            task_type = agent_config["task_type"]
            agent_name = agent_config["agent_name"]
            assignment = agent_config["assignment"]

            # -------------------------------------------------
            # Create persistent research task
            # -------------------------------------------------

            task_id = self.repository.create_task(
                project_id=project_id,
                task_type=task_type,
                description=assignment,
                assigned_agent=agent_name,
            )

            self.repository.update_task(
                task_id,
                "running",
            )

            # -------------------------------------------------
            # Create persistent agent run
            # -------------------------------------------------

            run_id = self.repository.create_agent_run(
                project_id=project_id,
                agent_name=agent_name,
                task=assignment,
                model="llama-3.1-8b-instant",
            )

            # -------------------------------------------------
            # Build specialist prompt
            # -------------------------------------------------

            prompt = f"""
You are the {agent_name}.

Here is research collected by the primary researcher:

{research[:6000]}

Your assignment:

{assignment}

Analyze ONLY the information provided above.

Do not invent facts.

Clearly identify uncertainty when the research does not
provide enough evidence.

Return concise findings in bullet points.
"""

            try:

                # -------------------------------------------------
                # Run specialist asynchronously
                # -------------------------------------------------

                result = await asyncio.to_thread(
                    self.specialist.run,
                    prompt,
                    False,
                )

                # -------------------------------------------------
                # Save successful agent run
                # -------------------------------------------------

                self.repository.complete_agent_run(
                    run_id,
                    result,
                )

                # -------------------------------------------------
                # Save successful task
                # -------------------------------------------------

                self.repository.update_task(
                    task_id,
                    "completed",
                    result,
                )

                return {
                    "agent_name": agent_name,
                    "task_id": task_id,
                    "run_id": run_id,
                    "result": result,
                    "status": "completed",
                }

            except Exception as error:

                # -------------------------------------------------
                # Save failed agent run
                # -------------------------------------------------

                self.repository.fail_agent_run(
                    run_id,
                    error,
                )

                # -------------------------------------------------
                # Save failed task
                # -------------------------------------------------

                self.repository.update_task(
                    task_id,
                    "failed",
                    str(error),
                )

                return {
                    "agent_name": agent_name,
                    "task_id": task_id,
                    "run_id": run_id,
                    "result": "",
                    "status": "failed",
                    "error": str(error),
                }

        # ---------------------------------------------------------
        # Run ALL specialist agents concurrently
        # ---------------------------------------------------------

        results = await asyncio.gather(
            *(
                run_agent(agent_config)
                for agent_config in tasks
            )
        )

        return results

    # =========================================================
    # CREATE DRAFT
    # =========================================================

    def create_draft(
        self,
        research,
        specialist_results,
    ):

        # Convert specialist dictionaries into readable text
        combined_sections = []

        for result in specialist_results:

            agent_name = result.get(
                "agent_name",
                "Unknown Specialist",
            )

            status = result.get(
                "status",
                "unknown",
            )

            findings = result.get(
                "result",
                "",
            )

            combined_sections.append(
                f"""
--- {agent_name} ---

Status: {status}

Findings:

{findings}
"""
            )

        combined = "\n\n".join(
            combined_sections
        )

        # ---------------------------------------------------------
        # Writer
        # ---------------------------------------------------------

        draft = self.writer.run(
            f"""
You are the lead technical writer.

Create a professional research report using ONLY the
information below.

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
6. Clearly identify claims that require verification.
7. Keep the report concise and professional.
8. Do not treat vendor claims as independently verified facts.
""",
            use_web=False,
        )

        return draft

    # =========================================================
    # REVIEW DRAFT
    # =========================================================

    def review_draft(
        self,
        research,
        draft,
    ):

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
9. Vendor claims presented as independently verified facts.

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

    def revise_draft(
        self,
        research,
        draft,
        review,
    ):

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
8. Do not present vendor claims as independently verified facts.
""",
            use_web=False,
        )

        return revised

    # =========================================================
    # MAIN SWARM
    # =========================================================

    async def run(
        self,
        topic,
    ):

        # =====================================================
        # CREATE PROJECT
        # =====================================================

        project_id = self.repository.create_project(
            topic
        )

        print(
            f"\nResearch project created: {project_id}"
        )

        # =====================================================
        # STEP 1: PRIMARY RESEARCH
        # =====================================================

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

        # -----------------------------------------------------
        # Create primary agent run
        # -----------------------------------------------------

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

Clearly distinguish:
- verified facts
- vendor/company claims
- uncertain information
""",
                use_web=True,
            )

            # -------------------------------------------------
            # Save successful primary agent run
            # -------------------------------------------------

            self.repository.complete_agent_run(
                research_run_id,
                research,
            )

            # -------------------------------------------------
            # Complete primary task
            # -------------------------------------------------

            self.repository.update_task(
                primary_task_id,
                "completed",
                research,
            )

        except Exception as error:

            # -------------------------------------------------
            # Save failure
            # -------------------------------------------------

            self.repository.fail_agent_run(
                research_run_id,
                error,
            )

            self.repository.update_task(
                primary_task_id,
                "failed",
                str(error),
            )

            self.repository.update_project_status(
                project_id,
                "failed",
            )

            raise

        # =====================================================
        # STEP 2: PARALLEL SPECIALIST AGENTS
        # =====================================================

        print(
            "[2/5] Parallel specialist agents working..."
        )

        tasks = [

            {
                "task_type": "technical_analysis",

                "agent_name": "TechnicalSpecialist",

                "assignment": """
Analyze this research from a technical/software engineering
perspective.

Identify:

- technologies
- tools
- architectures
- technical developments
- implementation approaches
- important engineering implications
""",
            },

            {
                "task_type": "market_analysis",

                "agent_name": "MarketSpecialist",

                "assignment": """
Analyze this research from a market and industry perspective.

Identify:

- companies
- products
- adoption trends
- industry impact
- important market developments
- vendor claims that need verification
""",
            },

            {
                "task_type": "fact_checking",

                "agent_name": "FactChecker",

                "assignment": """
Analyze this research as a skeptical fact-checker.

Identify:

- claims requiring verification
- possible inconsistencies
- weak evidence
- unsupported conclusions
- statistics that need verification
- claims that appear to come only from vendors
""",
            },
        ]

        specialist_results = await self.run_parallel_agents(
            project_id,
            research,
            tasks,
        )

        # =====================================================
        # CHECK SPECIALIST RESULTS
        # =====================================================

        successful_specialists = [
            result
            for result in specialist_results
            if result.get("status") == "completed"
        ]

        failed_specialists = [
            result
            for result in specialist_results
            if result.get("status") == "failed"
        ]

        print(
            f"    Specialists completed: "
            f"{len(successful_specialists)}/{len(tasks)}"
        )

        if failed_specialists:

            print(
                f"    Specialist failures: "
                f"{len(failed_specialists)}"
            )

        # =====================================================
        # STEP 3: CREATE INITIAL DRAFT
        # =====================================================

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

        # =====================================================
        # STEP 4: QUALITY GATE
        # =====================================================

        max_revisions = 2

        for attempt in range(
            max_revisions + 1
        ):

            print(
                f"[4/5] Quality review "
                f"(attempt {attempt + 1})..."
            )

            review = self.review_draft(
                research,
                draft,
            )

            # -------------------------------------------------
            # Extract review status
            # -------------------------------------------------

            if isinstance(
                review,
                dict,
            ):

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
            # Extract review information
            # -------------------------------------------------

            if isinstance(
                review,
                dict,
            ):

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

                revision_notes = str(
                    review
                )

            # -------------------------------------------------
            # Save review
            # -------------------------------------------------

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

            # =================================================
            # PASS
            # =================================================

            if status == "PASS":

                self.repository.update_project_status(
                    project_id,
                    "completed",
                )

                print(
                    "[5/5] Quality gate passed."
                )

                return draft

            # =================================================
            # MAXIMUM REVISIONS
            # =================================================

            if attempt == max_revisions:

                self.repository.update_project_status(
                    project_id,
                    "completed_with_warnings",
                )

                print(
                    "[5/5] Maximum revisions reached."
                )

                return draft

            # =================================================
            # REVISION
            # =================================================

            print(
                f"    Revising draft "
                f"(revision {attempt + 1})..."
            )

            draft = self.revise_draft(
                research,
                draft,
                review,
            )

            # -------------------------------------------------
            # Save revised draft
            # -------------------------------------------------

            draft_id = self.repository.create_draft(
                project_id=project_id,
                content=draft,
                version=attempt + 2,
            )

        return draft