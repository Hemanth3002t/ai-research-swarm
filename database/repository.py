from database.connection import get_connection


class ResearchRepository:

    # =========================================================
    # PROJECTS
    # =========================================================

    def create_project(self, topic):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO projects (topic, status)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (topic, "running"),
                )

                project_id = cursor.fetchone()[0]

            connection.commit()

            return project_id

        finally:
            connection.close()

    def update_project_status(self, project_id, status):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE projects
                    SET status = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (status, project_id),
                )

            connection.commit()

        finally:
            connection.close()

    # =========================================================
    # RESEARCH TASKS
    # =========================================================

    def create_task(
        self,
        project_id,
        task_type,
        description,
        assigned_agent=None,
    ):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO research_tasks
                    (
                        project_id,
                        task_type,
                        description,
                        assigned_agent
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        project_id,
                        task_type,
                        description,
                        assigned_agent,
                    ),
                )

                task_id = cursor.fetchone()[0]

            connection.commit()

            return task_id

        finally:
            connection.close()

    def update_task(
        self,
        task_id,
        status,
        result=None,
    ):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:

                if status == "running":
                    cursor.execute(
                        """
                        UPDATE research_tasks
                        SET status = %s,
                            started_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (status, task_id),
                    )

                elif status == "completed":
                    cursor.execute(
                        """
                        UPDATE research_tasks
                        SET status = %s,
                            result = %s,
                            completed_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (
                            status,
                            result,
                            task_id,
                        ),
                    )

                else:
                    cursor.execute(
                        """
                        UPDATE research_tasks
                        SET status = %s,
                            result = %s
                        WHERE id = %s
                        """,
                        (
                            status,
                            result,
                            task_id,
                        ),
                    )

            connection.commit()

        finally:
            connection.close()

    # =========================================================
    # AGENT RUNS
    # =========================================================

    def create_agent_run(
        self,
        project_id,
        agent_name,
        task,
        model,
    ):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO agent_runs
                    (
                        project_id,
                        agent_name,
                        task,
                        model,
                        status
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        project_id,
                        agent_name,
                        task,
                        model,
                        "started",
                    ),
                )

                run_id = cursor.fetchone()[0]

            connection.commit()

            return run_id

        finally:
            connection.close()

    def complete_agent_run(
        self,
        run_id,
        result,
        input_tokens=0,
        output_tokens=0,
    ):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE agent_runs
                    SET status = %s,
                        result = %s,
                        input_tokens = %s,
                        output_tokens = %s,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (
                        "completed",
                        result,
                        input_tokens,
                        output_tokens,
                        run_id,
                    ),
                )

            connection.commit()

        finally:
            connection.close()

    def fail_agent_run(
        self,
        run_id,
        error,
    ):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE agent_runs
                    SET status = %s,
                        error = %s,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (
                        "failed",
                        str(error),
                        run_id,
                    ),
                )

            connection.commit()

        finally:
            connection.close()

    # =========================================================
    # DRAFTS
    # =========================================================

    def create_draft(
        self,
        project_id,
        content,
        version=1,
    ):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO drafts
                    (
                        project_id,
                        version,
                        content
                    )
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (
                        project_id,
                        version,
                        content,
                    ),
                )

                draft_id = cursor.fetchone()[0]

            connection.commit()

            return draft_id

        finally:
            connection.close()

    # =========================================================
    # REVIEWS
    # =========================================================

    def create_review(
        self,
        project_id,
        draft_id,
        status,
        critical_flaws,
        revision_notes,
    ):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO reviews
                    (
                        project_id,
                        draft_id,
                        status,
                        critical_flaws,
                        revision_notes
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        project_id,
                        draft_id,
                        status,
                        critical_flaws,
                        revision_notes,
                    ),
                )

                review_id = cursor.fetchone()[0]

            connection.commit()

            return review_id

        finally:
            connection.close()

    # =========================================================
    # RESEARCH SOURCES
    # =========================================================

    def create_source(
        self,
        project_id,
        title,
        url,
        content,
        source_type="web",
    ):
        connection = get_connection()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO research_sources
                    (
                        project_id,
                        title,
                        url,
                        content,
                        source_type
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        project_id,
                        title,
                        url,
                        content,
                        source_type,
                    ),
                )

                source_id = cursor.fetchone()[0]

            connection.commit()

            return source_id

        finally:
            connection.close()