from database.repository import ResearchRepository


def main():

    repository = ResearchRepository()

    # Create test project
    project_id = repository.create_project(
        "Database integration test"
    )

    print(f"Project created: {project_id}")

    # Create test task
    task_id = repository.create_task(
        project_id=project_id,
        task_type="test",
        description="Testing database repository",
        assigned_agent="TestAgent",
    )

    print(f"Task created: {task_id}")

    # Mark task running
    repository.update_task(
        task_id,
        "running",
    )

    print("Task marked as running.")

    # Complete task
    repository.update_task(
        task_id,
        "completed",
        "Database repository test successful.",
    )

    print("Task completed.")

    # Update project
    repository.update_project_status(
        project_id,
        "completed",
    )

    print("Project completed.")

    print("\nRepository test successful!")


if __name__ == "__main__":
    main()