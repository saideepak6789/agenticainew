# Simple Document Generator using CrewAI

from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv(override=True)

# 1. Create Agents
researcher = Agent(
    role="Tech Researcher",
    goal="Research and gather accurate information about technologies",
    backstory="You are an expert at finding reliable technical information and summarizing key concepts.",
    verbose=True
)

writer = Agent(
    role="Technical Writer", 
    goal="Create clear, structured documentation from research",
    backstory="You excel at writing technical documentation that is easy to understand and well-organized.",
    verbose=True
)

# 2. Define Tasks
research_task = Task(
    description="Research {technology} and gather information about:\n"
                "- What it is and main purpose\n"
                "- Key features and benefits\n" 
                "- Basic usage/implementation\n"
                "- Common use cases",
    expected_output="A comprehensive research summary with key points about the technology",
    agent=researcher
)

documentation_task = Task(
    description="Using the research findings, write technical documentation for {technology} that includes:\n"
                "- Overview section\n"
                "- Key Features\n"
                "- Getting Started guide\n" 
                "- Use Cases\n"
                "Format as clean markdown with proper headings",
    expected_output="Complete technical documentation in markdown format, ready for publication",
    agent=writer
)

# 3. Create Crew
doc_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, documentation_task],
    verbose=True
)

# 4. Run the crew
if __name__ == "__main__":
    # Change this to any technology you want to document
    technology = "Docker"
    
    print(f"Starting documentation generation for: {technology}")
    print("=" * 50)
    
    result = doc_crew.kickoff(inputs={"technology": technology})
    
    print("\n" + "=" * 50)
    print("GENERATED DOCUMENTATION:")
    print("=" * 50)
    print(result)
    
    # Optional: Save to file
    with open(f"{technology.lower()}_documentation.md", "w") as f:
        f.write(str(result))
    print(f"\nDocumentation saved to: {technology.lower()}_documentation.md")