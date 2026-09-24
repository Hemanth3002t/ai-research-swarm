import asyncio
import re
from urllib.parse import urlparse

from agents.agent import ResearchAgent
from database.repository import ResearchRepository


class ResearchSwarm:

    def __init__(self):
        self.repository = ResearchRepository()

        # =====================================================
        # MODELS
        # =====================================================
        #
        # Use GPT-OSS only for the main research task.
        # All agents use the confirmed Groq model to keep model selection consistent.
        #

        self.researcher = ResearchAgent(
            model="openai/gpt-oss-20b",
            max_completion_tokens=450,
        )

        self.specialist = ResearchAgent(
            model="openai/gpt-oss-20b",
            max_completion_tokens=300,
        )

        self.writer = ResearchAgent(
            model="openai/gpt-oss-20b",
            max_completion_tokens=450,
        )

        self.critic = ResearchAgent(
            model="openai/gpt-oss-20b",
            max_completion_tokens=350,
        )

    # =========================================================
    # URL VALIDATION
    # =========================================================

    @staticmethod
    def is_valid_url(url):
        if not isinstance(url, str):
            return False

        url = url.strip()

        if not url:
            return False

        try:
            parsed = urlparse(url)

            return (
                parsed.scheme in ("http", "https")
                and bool(parsed.netloc)
            )

        except Exception:
            return False

    # =========================================================
    # EXTRACT URLS FROM RAW RESEARCH
    # =========================================================

    @staticmethod
    def extract_urls(text):
        """
        Extract literal HTTP/HTTPS URLs from researcher output.

        This is a deterministic fallback. It does not depend
        on the LLM correctly formatting the sources.
        """

        if not text:
            return []

        pattern = r"https?://[^\s<>\]\[\"')]+"

        matches = re.findall(
            pattern,
            str(text),
            flags=re.IGNORECASE,
        )

        cleaned = []

        for url in matches:

            url = url.rstrip(
                ".,;:!?)]}\"'"
            )

            if url not in cleaned:
                cleaned.append(url)

        return cleaned

    # =========================================================
    # SAVE SOURCES
    # =========================================================

    def save_research_sources(
        self,
        project_id,
        sources,
    ):

        if not isinstance(sources, list):
            print("    No valid source list returned.")
            return 0

        saved_count = 0
        seen_urls = set()

        for source in sources:

            if not isinstance(source, dict):
                continue

            title = str(
                source.get("title", "")
            ).strip()

            url = str(
                source.get("url", "")
            ).strip()

            content = str(
                source.get("content", "")
            ).strip()

            source_type = str(
                source.get(
                    "source_type",
                    "web",
                )
            ).strip()

            if not title:
                title = "Untitled source"

            if not content:
                content = (
                    "No source summary was provided."
                )

            if not self.is_valid_url(url):
                print(
                    f"    Skipping invalid source URL: {url}"
                )
                continue

            # Prevent duplicate URLs.
            normalized_url = url.lower().rstrip("/")

            if normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)

            try:

                source_id = self.repository.create_source(
                    project_id=project_id,
                    title=title,
                    url=url,
                    content=content,
                    source_type=source_type,
                )

                saved_count += 1

                print(
                    f"    Source saved: "
                    f"{source_id} - {title}"
                )

            except Exception as error:

                print(
                    f"    Failed to save source "
                    f"'{title}': {error}"
                )

        return saved_count

    # =====================================================
    # EXTRACT AND SAVE SOURCES
    # =====================================================

    def extract_and_save_sources(
        self,
        project_id,
        research,
    ):
        """
        Extract literal URLs from the primary research text
        and save them directly to PostgreSQL.
        """

        url_pattern = r'https?://[^\s<>"\]\)]+'
        urls = re.findall(
            url_pattern,
            research,
        )

        unique_urls = list(
            dict.fromkeys(urls)
        )

        print(
            f"    Literal URLs found: "
            f"{len(unique_urls)}"
        )

        saved = 0

        for url in unique_urls:

            url = url.rstrip(
                ".,;:!?)]}"
            )

            try:

                self.repository.create_source(
                    project_id=project_id,
                    title=url,
                    url=url,
                    content=(
                        "URL extracted from primary research."
                    ),
                    source_type="web",
                )

                saved += 1

            except Exception as error:

                print(
                    f"    Failed to save source: {url}"
                )

                print(
                    f"    Error: {error}"
                )

        print(
            f"    Sources saved: {saved}"
        )

        return unique_urls

    # =========================================================
    # PARALLEL SPECIALISTS
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

            run_id = self.repository.create_agent_run(
                project_id=project_id,
                agent_name=agent_name,
                task=assignment,
                model=self.specialist.model,
            )

            prompt = f"""
You are the {agent_name}.

Research summary:

{research[:4500]}

Assignment:

{assignment}

Rules:
- Use only the supplied research.
- Do not invent facts.
- Flag uncertainty.
- Keep the answer concise.
- Use bullet points.
"""

            try:

                result = await asyncio.to_thread(
                    self.specialist.run,
                    prompt,
                    False,
                )

                self.repository.complete_agent_run(
                    run_id,
                    result,
                )

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

            except Exception as first_error:

                print(
                     f"    {agent_name} failed: "
                     f"{type(first_error).__name__}: {first_error}"
                      )
                print(
                     f"    {agent_name} failed. "
                     f"Retrying once..."
                     )

                # Retry with a smaller prompt.
                try:

                    retry_prompt = f"""
Analyze the research below.

Research:
{research[:3000]}

Task:
{assignment}

Return only concise bullet points.
Do not invent facts.
"""

                    result = await asyncio.to_thread(
                        self.specialist.run,
                        retry_prompt,
                        False,
                    )

                    self.repository.complete_agent_run(
                        run_id,
                        result,
                    )

                    self.repository.update_task(
                        task_id,
                        "completed",
                        result,
                    )

                    print(
                        f"    {agent_name} retry succeeded."
                    )

                    return {
                        "agent_name": agent_name,
                        "task_id": task_id,
                        "run_id": run_id,
                        "result": result,
                        "status": "completed",
                    }

                except Exception as second_error:

                    print(
                        f"    {agent_name} retry failed: "
                        f"{type(second_error).__name__}: {second_error}"
                    )

                    error_text = (
                        f"Initial error: {first_error}\n"
                        f"Retry error: {second_error}"
                    )

                    self.repository.fail_agent_run(
                        run_id,
                        second_error,
                    )

                    self.repository.update_task(
                        task_id,
                        "failed",
                        error_text,
                    )

                    return {
                        "agent_name": agent_name,
                        "task_id": task_id,
                        "run_id": run_id,
                        "result": "",
                        "status": "failed",
                        "error": error_text,
                    }

        return await asyncio.gather(
            *(
                run_agent(agent_config)
                for agent_config in tasks
            )
        )

    # =========================================================
    # CREATE DRAFT
    # =========================================================

    def create_draft(
        self,
        project_id,
        research,
        specialist_results,
        version=1,
        task="Create research report draft",
    ):

        combined_sections = []

        for result in specialist_results:

            combined_sections.append(
                f"""
--- {result.get("agent_name", "Unknown Specialist")} ---

Status:
{result.get("status", "unknown")}

Findings:
{result.get("result", "")}
"""
            )

        combined = "\n\n".join(
            combined_sections
        )

        task_id = self.repository.create_task(
            project_id=project_id,
            task_type="draft_creation",
            description=task,
            assigned_agent="Writer",
        )

        self.repository.update_task(
            task_id,
            "running",
        )

        run_id = self.repository.create_agent_run(
            project_id=project_id,
            agent_name="Writer",
            task=task,
            model=self.writer.model,
        )

        try:

            draft = self.writer.run(
                f"""
You are the lead technical writer.

Create a concise professional research report.

PRIMARY RESEARCH:
{research[:4500]}

SPECIALIST FINDINGS:
{combined[:6000]}

Rules:

1. Use clear headings.
2. Separate facts from uncertain claims.
3. Do not invent information.
4. Do not create unsupported statistics.
5. Clearly identify vendor claims.
6. Keep it concise.

OUTPUT RULES:
Return ONLY the final research report.
Do not include:
- internal reasoning
- analysis
- planning
- checklists
- confidence scores
- mental sandbox
- instructions to yourself
- <think> tags
- <analysis> tags
- commentary about the prompt
Start directly with the report title.
End directly after the report.
Do not describe how you generated the report.
""",
                use_web=False,
            )

            self.repository.complete_agent_run(
                run_id,
                draft,
            )

            self.repository.update_task(
                task_id,
                "completed",
                draft,
            )

            draft_id = self.repository.create_draft(
                project_id=project_id,
                content=draft,
                version=version,
            )

            return draft, draft_id

        except Exception as error:

            self.repository.fail_agent_run(
                run_id,
                error,
            )

            self.repository.update_task(
                task_id,
                "failed",
                str(error),
            )

            raise

    # =========================================================
    # REVIEW
    # =========================================================

    def review_draft(
        self,
        project_id,
        research,
        draft,
        task="Review research report draft",
    ):

        task_id = self.repository.create_task(
            project_id=project_id,
            task_type="draft_review",
            description=task,
            assigned_agent="Critic",
        )

        self.repository.update_task(
            task_id,
            "running",
        )

        run_id = self.repository.create_agent_run(
            project_id=project_id,
            agent_name="Critic",
            task=task,
            model=self.critic.model,
        )

        try:

            review = self.critic.run(
                f"""
You are a strict research quality reviewer.

SOURCE RESEARCH:
{research[:4000]}

DRAFT:
{draft[:5000]}

Check for:

- unsupported claims
- factual contradictions
- invented statistics
- misleading conclusions
- vendor claims presented as facts
- claims requiring evidence

Return ONLY valid JSON:

{{
  "status": "PASS",
  "critical_flaws": [],
  "revision_notes": ""
}}

OR:

{{
  "status": "FAIL",
  "critical_flaws": ["specific issue"],
  "revision_notes": "Specific revision instructions."
}}

Rules:
- PASS only if there are no significant problems.
- FAIL if significant problems exist.
- critical_flaws must contain short items.
- revision_notes must be under 80 words.
""",
                use_web=False,
                json_mode=True,
            )

            self.repository.complete_agent_run(
                run_id,
                str(review),
            )

            self.repository.update_task(
                task_id,
                "completed",
                str(review),
            )

            return review

        except Exception as error:

            self.repository.fail_agent_run(
                run_id,
                error,
            )

            self.repository.update_task(
                task_id,
                "failed",
                str(error),
            )

            return {
                "status": "FAIL",
                "critical_flaws": [
                    "Quality reviewer failed."
                ],
                "revision_notes": (
                    "Retry the quality review."
                ),
            }

    # =========================================================
    # REVISE DRAFT
    # =========================================================

    def revise_draft(
        self,
        project_id,
        research,
        draft,
        review,
        version,
    ):

        if isinstance(review, dict):

            review_text = (
                f"Status: "
                f"{review.get('status', 'FAIL')}\n"
                f"Critical flaws: "
                f"{review.get('critical_flaws', [])}\n"
                f"Revision notes: "
                f"{review.get('revision_notes', '')}"
            )

        else:

            review_text = str(review)

        task = (
            f"Revise research draft "
            f"version {version}"
        )

        task_id = self.repository.create_task(
            project_id=project_id,
            task_type="draft_revision",
            description=task,
            assigned_agent="TechnicalWriter",
        )

        self.repository.update_task(
            task_id,
            "running",
        )

        run_id = self.repository.create_agent_run(
            project_id=project_id,
            agent_name="TechnicalWriter",
            task=task,
            model=self.writer.model,
        )

        try:

            revised = self.writer.run(
                f"""
Revise this research report.

SOURCE RESEARCH:
{research[:4500]}

CURRENT DRAFT:
{draft[:5500]}

QUALITY REVIEW:
{review_text}

Rules:

1. Fix every issue identified.
2. Remove unsupported claims.
3. Preserve useful information.
4. Do not invent replacement facts.
5. Clearly qualify uncertain claims.
6. Keep the report concise.

OUTPUT RULES:
Return ONLY the final research report.
Do not include:
- internal reasoning
- analysis
- planning
- checklists
- confidence scores
- mental sandbox
- instructions to yourself
- <think> tags
- <analysis> tags
- commentary about the prompt
Start directly with the report title.
End directly after the report.
Do not describe how you generated the report.
""",
                use_web=False,
            )

            self.repository.complete_agent_run(
                run_id,
                revised,
            )

            self.repository.update_task(
                task_id,
                "completed",
                revised,
            )

            draft_id = self.repository.create_draft(
                project_id=project_id,
                content=revised,
                version=version,
            )

            return revised, draft_id

        except Exception as error:

            self.repository.fail_agent_run(
                run_id,
                error,
            )

            self.repository.update_task(
                task_id,
                "failed",
                str(error),
            )

            raise

    # =========================================================
    # MAIN SWARM
    # =========================================================

    async def run(self, topic):

        project_id = self.repository.create_project(
            topic
        )

        print(
            f"\nResearch project created: "
            f"{project_id}"
        )

        # =====================================================
        # STEP 1 — PRIMARY RESEARCH
        # =====================================================

        print(
            "\n[1/6] Primary researcher working..."
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

        research_run_id = (
            self.repository.create_agent_run(
                project_id=project_id,
                agent_name="PrimaryResearcher",
                task=topic,
                model=self.researcher.model,
            )
        )

        try:

            raw_research = self.researcher.run(
                f"""
Research this topic using current web sources:

{topic}

Collect:

- important facts
- recent developments
- companies and tools
- relevant dates
- evidence
- source names
- EXACT source URLs

IMPORTANT:

For every important web source, include a literal URL.

Use this format:

SOURCE:
Title: Example
URL: https://example.com
Type: news
Evidence: Short evidence summary

Clearly distinguish:

- verified facts
- vendor claims
- uncertain information

Do not invent facts or sources.

Keep the research concise.
""",
                use_web=True,
                json_mode=False,
            )

            if not raw_research:
                raise ValueError(
                    "Primary researcher returned empty research."
                )

            raw_research = str(
                raw_research
            ).strip()

            self.repository.complete_agent_run(
                research_run_id,
                raw_research,
            )

            self.repository.update_task(
                primary_task_id,
                "completed",
                raw_research,
            )

        except Exception as error:

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
        # STEP 2 — SOURCE EXTRACTION
        # =====================================================

        print(
            "[2/6] Extracting and saving research sources..."
        )

        source_urls = (
            self.extract_and_save_sources(
                project_id,
                raw_research,
            )
        )

        # =====================================================
        # STEP 3 — SPECIALISTS
        # =====================================================

        print(
            "[3/6] Parallel specialist agents working..."
        )

        tasks = [

            {
                "task_type": "technical_analysis",
                "agent_name": "TechnicalSpecialist",
                "assignment": """
Analyze the research from a technical perspective.

Identify:
- technologies
- tools
- architectures
- technical developments
- engineering implications
""",
            },

            {
                "task_type": "market_analysis",
                "agent_name": "MarketSpecialist",
                "assignment": """
Analyze the research from a market perspective.

Identify:
- companies
- products
- adoption trends
- industry impact
- vendor claims
""",
            },

            {
                "task_type": "fact_checking",
                "agent_name": "FactChecker",
                "assignment": """
Analyze the research as a skeptical fact checker.

Identify:
- claims needing verification
- weak evidence
- unsupported conclusions
- questionable statistics
- vendor-only claims
""",
            },

        ]

        specialist_results = (
            await self.run_parallel_agents(
                project_id,
                raw_research,
                tasks,
            )
        )

        successful_specialists = [
            result
            for result in specialist_results
            if result.get("status")
            == "completed"
        ]

        failed_specialists = [
            result
            for result in specialist_results
            if result.get("status")
            == "failed"
        ]

        print(
            f"    Specialists completed: "
            f"{len(successful_specialists)}/"
            f"{len(tasks)}"
        )

        if failed_specialists:

            print(
                f"    Specialist failures: "
                f"{len(failed_specialists)}"
            )

        # =====================================================
        # STEP 4 — WRITER
        # =====================================================

        print(
            "[4/6] Writer creating draft..."
        )

        draft, draft_id = (
            self.create_draft(
                project_id,
                raw_research,
                specialist_results,
                version=1,
            )
        )

        # =====================================================
        # STEP 5 — QUALITY GATE
        # =====================================================

        max_revisions = 2
        best_draft = draft
        best_draft_id = draft_id
        best_score = None

        for attempt in range(
            max_revisions + 1
        ):

            print(
                f"[5/6] Quality review "
                f"(attempt {attempt + 1})..."
            )

            review = self.review_draft(
                project_id,
                raw_research,
                draft,
            )

            if isinstance(
                review,
                dict
            ):

                status = review.get(
                    "status",
                    "FAIL",
                )

                critical_flaws = (
                    review.get(
                        "critical_flaws",
                        [],
                    )
                )

                revision_notes = (
                    review.get(
                        "revision_notes",
                        "",
                    )
                )

            else:

                status = "FAIL"

                critical_flaws = [
                    "Reviewer returned invalid data."
                ]

                revision_notes = str(
                    review
                )

            status = str(
                status
            ).upper().strip()

            print(
                f"    Review result: "
                f"{status}"
            )

            if critical_flaws:
                print("    Critical flaws:")
                print(f"      {critical_flaws}")

            if revision_notes:
                print("    Revision notes:")
                print(f"      {revision_notes}")

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

            if isinstance(
                critical_flaws,
                list,
            ):
                flaw_count = len(
                    critical_flaws
                )
            elif critical_flaws:
                flaw_count = 1
            else:
                flaw_count = 0

            review_score = (
                0 if status == "PASS" else 1,
                flaw_count,
            )

            if (
                best_score is None
                or review_score < best_score
            ):
                best_score = review_score
                best_draft = draft
                best_draft_id = draft_id

            elif attempt > 0:

                self.repository.update_project_status(
                    project_id,
                    "completed_with_warnings",
                )

                print(
                    "[6/6] Revision did not improve "
                    "the quality score. Returning the "
                    "best available draft with warnings."
                )

                return best_draft

            # =================================================
            # PASS
            # =================================================

            if status == "PASS":

                if failed_specialists:

                    self.repository.update_project_status(
                        project_id,
                        "completed_with_warnings",
                    )

                    print(
                        "[6/6] Quality gate passed "
                        "with specialist warnings."
                    )

                else:

                    self.repository.update_project_status(
                        project_id,
                        "completed",
                    )

                    print(
                        "[6/6] Quality gate passed."
                    )

                return draft

            # =================================================
            # MAX REVISIONS
            # =================================================

            if attempt == max_revisions:

                self.repository.update_project_status(
                    project_id,
                    "completed_with_warnings",
                )

                print(
                    "[6/6] Maximum revisions reached. "
                    "Returning best available draft "
                    "with warnings."
                )

                return best_draft

            # =================================================
            # REVISION
            # =================================================

            print(
                f"    Revising draft "
                f"(revision {attempt + 1})..."
            )

            draft, draft_id = (
                self.revise_draft(
                    project_id,
                    raw_research,
                    draft,
                    review,
                    version=attempt + 2,
                )
            )

        return draft
