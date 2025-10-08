# Log Analyzer using CrewAI

import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

load_dotenv(override=True)

# Fetch logs
url = "https://raw.githubusercontent.com/logpai/loghub/master/Spark/Spark_2k.log"
logs = requests.get(url).text[:2000]  # First 2000 chars

# Agents
parser = Agent(role="Log Parser", goal="Parse Spark logs", backstory="Spark log expert")
analyst = Agent(role="Analyst", goal="Find issues", backstory="Performance expert")

# Tasks
parse_task = Task(
    description=f"Parse these Spark logs:\n{logs}",
    expected_output="Summary of log patterns and errors",
    agent=parser
)
analyze_task = Task(
    description="Identify top 3 issues and recommendations",
    expected_output="Critical issues and fixes",
    agent=analyst
)

# Run
crew = Crew(agents=[parser, analyst], tasks=[parse_task, analyze_task])
result = crew.kickoff()
print(result)