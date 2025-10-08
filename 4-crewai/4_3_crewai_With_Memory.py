import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv(override=True)

# ----------------------------
# Fetch sample files from repo
# ----------------------------
def get_file(path, limit=1500):
    url = f"https://raw.githubusercontent.com/fazt/chat-javascript-fullstack/main/{path}"
    try:
        return requests.get(url).text[:limit]
    except:
        return "File not found"

frontend_code = get_file("src/index.js")
backend_code = get_file("src/server/index.js")
package_json = get_file("package.json", limit=800)

# ----------------------------
# Setup memory per agent
# ----------------------------
# Each is a dictionary with keys "previous_issues" and "recommendations"
# Each key contains a list of strings/values
security_memory = {"previous_issues": [], "recommendations": []}
quality_memory = {"quality_issues": [], "recommendations": []}
architecture_memory = {"architecture_notes": []}

# ----------------------------
# Agents
# ----------------------------
security_reviewer = Agent(
    role="Security Reviewer",
    goal="Find security vulnerabilities",
    backstory="Expert at identifying security issues in JavaScript code",
    memory=security_memory,
    verbose=True
)

quality_reviewer = Agent(
    role="Code Quality Reviewer",
    goal="Assess code quality and best practices",
    backstory="Expert at JavaScript best practices and clean code",
    memory=quality_memory,
    verbose=True
)

architect = Agent(
    role="Architecture Reviewer",
    goal="Evaluate overall project structure",
    backstory="Full-stack architecture expert",
    memory=architecture_memory,
    verbose=True
)

# ----------------------------
# Tasks
# ----------------------------
security_task = Task(
    description=f"Review frontend and backend code for security issues:\nFrontend:\n{frontend_code}\nBackend:\n{backend_code}",
    expected_output="List of security vulnerabilities and suggested fixes",
    agent=security_reviewer
)

quality_task = Task(
    description=f"Review code quality:\nFrontend:\n{frontend_code}\nBackend:\n{backend_code}",
    expected_output="Code quality issues and improvements",
    agent=quality_reviewer,
    context=[security_task]  # Can see security review results
)

architecture_task = Task(
    description=f"Review project architecture:\nPackage.json:\n{package_json}\nCode structure observed",
    expected_output="Architecture assessment and recommendations",
    agent=architect,
    context=[security_task, quality_task]  # Can see previous results
)

# ----------------------------
# Run Crew
# ----------------------------
crew = Crew(
    agents=[security_reviewer, quality_reviewer, architect],
    tasks=[security_task, quality_task, architecture_task],
    verbose=True
)

print("Starting code review of chat-javascript-fullstack...\n")
result = crew.kickoff()

print("\nCODE REVIEW RESULTS:")
print("=" * 50)
print(result)
