import asyncio

from agents.orchestrator import ResearchSwarm


async def main():
    swarm = ResearchSwarm()

    result = await swarm.run(
        "The current state of AI-powered software testing in 2026"
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESEARCH REPORT")
    print("=" * 70)
    print("\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())