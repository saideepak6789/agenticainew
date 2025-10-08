# Code Reviewer using CrewAI
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv(override=True)

# Fetch sample files from the repository
def get_file(path):
    url = f"https://raw.githubusercontent.com/fazt/chat-javascript-fullstack/main/{path}"
    try:
        return requests.get(url).text
    except:
        return "File not found"

# Get key files for review
frontend_code = get_file("src/index.js")[:1500]  # Limit size
backend_code = get_file("src/server/index.js")[:1500]
package_json = get_file("package.json")[:800]

# Agents
security_reviewer = Agent(
    role="Security Reviewer", 
    goal="Find security vulnerabilities",
    backstory="Expert at identifying security issues in JavaScript code"
)

quality_reviewer = Agent(
    role="Code Quality Reviewer",
    goal="Assess code quality and best practices", 
    backstory="Expert at JavaScript best practices and clean code"
)

architect = Agent(
    role="Architecture Reviewer",
    goal="Evaluate overall project structure",
    backstory="Full-stack architecture expert"
)

# Tasks
security_task = Task(
    description=f"Review for security issues:\nFrontend:\n{frontend_code}\nBackend:\n{backend_code}",
    expected_output="List of security vulnerabilities and fixes",
    agent=security_reviewer
)

quality_task = Task(
    description=f"Review code quality:\nFrontend:\n{frontend_code}\nBackend:\n{backend_code}",
    expected_output="Code quality issues and improvements",
    agent=quality_reviewer
)

architecture_task = Task(
    description=f"Review project structure:\nPackage.json:\n{package_json}\nCode structure observed",
    expected_output="Architecture assessment and recommendations", 
    agent=architect
)

# Run review
crew = Crew(
    agents=[security_reviewer, quality_reviewer, architect],
    tasks=[security_task, quality_task, architecture_task]
)

print("Starting code review of chat-javascript-fullstack...")
result = crew.kickoff()
print("\nCODE REVIEW RESULTS:")
print("=" * 50)
print(result)