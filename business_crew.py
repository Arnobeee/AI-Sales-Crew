from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime # NEW: To handle the timestamp

# 1. Connect to local Ollama
local_llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

# 2. Define the Search Tool properly
@tool("internet_search")
def internet_search(query: str):
    """Searches the internet for the latest information on a topic."""
    return DuckDuckGoSearchRun().run(query)

# 3. Define the Researcher Agent
researcher = Agent(
    role='Business Case Researcher',
    goal='Find 3 current trends in AI automation for accounting firms in 2026.',
    backstory='You are a tech-savvy researcher specialized in emerging business technology.',
    llm=local_llm,
    verbose=True,
    tools=[internet_search],
    allow_delegation=False
)

# 4. Define the Sales Writer Agent
writer = Agent(
    role='Expert Sales Copywriter',
    goal='Write a friendly 3-paragraph email to a business owner based on the trends found.',
    backstory='You turn complex AI research into clear, persuasive business emails.',
    llm=local_llm,
    verbose=True,
    allow_delegation=False
)

# 5. Define the Tasks
research_task = Task(
    description='Use the search tool to find 3 ways AI is specifically helping accountants in 2026.',
    expected_output='A summary of 3 distinct AI trends in accounting.',
    agent=researcher
)

writing_task = Task(
    description="Using the researcher's findings, write a compelling cold email to an accounting firm owner.",
    expected_output='A polished 3-paragraph email ready to be sent.',
    agent=writer,
    context=[research_task]
)

# 6. Form the Crew
sales_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

print("### Starting the Business Crew...")
result = sales_crew.kickoff()

# --- NEW: MANUAL SAVE WITH TIMESTAMP ---
# Generate a timestamp (YearMonthDay_HourMinute)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
filename = f"accounting_email_{timestamp}.md"

# Force Python to write the file
with open(filename, "w", encoding="utf-8") as f:
    f.write(str(result))

print(f"\n✔ SUCCESS: Result saved as: {filename}")