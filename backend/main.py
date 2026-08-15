from agents.agent import ResearchAgent

def main():
    agent = ResearchAgent()

    result = agent.run(
        "Explain the major applications of artificial intelligence "
        "in software testing."
    )

    print("\n===== RESEARCH AGENT =====\n")
    print(result)


if __name__ == "__main__":
    main()