from crewai import Agent, Task, Crew, Process,LLM
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from tools.yf_tech_analysis import YFinanceTechnicalAnalysisTool
from tools.yf_fundamental_analysis import YFinanceFundamentalAnalysisTool
from tools.sentiment_analysis import RedditSentimentAnalysisTool
from tools.yahoo_finance_tool import YahooFinanceNewsTool
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import yaml


# Environment Variables
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["REDDIT_CLIENT_ID"] = os.getenv("REDDIT_CLIENT_ID")
os.environ["REDDIT_CLIENT_SECRET"] = os.getenv("REDDIT_CLIENT_SECRET")
os.environ["REDDIT_USER_AGENT"] = os.getenv("REDDIT_USER_AGENT")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")
# Updated to use the current supported model
MODEL_NAME = "llama-3.3-70b-versatile"  # Updated from llama-3.1-70b-versatile

llm = LLM(model="groq/llama3-8b-8192"   api_key=groq_api_key  )  # No need to specify the API key, it picks up from the environment

# Fix the path to yaml files - check if they're in the config directory, otherwise use root
config_paths = [
    ('config/agents.yaml', 'config/tasks.yaml'),
    ('agents.yaml', 'tasks.yaml')
]

agents_config = None
tasks_config = None

for agents_path, tasks_path in config_paths:
    try:
        with open(agents_path, 'r') as file:
            agents_config = yaml.safe_load(file)
        with open(tasks_path, 'r') as file:
            tasks_config = yaml.safe_load(file)
        print(f"Loaded configuration from {agents_path} and {tasks_path}")
        break
    except FileNotFoundError:
        continue

if agents_config is None or tasks_config is None:
    raise FileNotFoundError("Could not find agents.yaml and tasks.yaml in config/ or root directory")

class ResearchReport(BaseModel):
    researchreport: str

class TechnicalAnalysisReport(BaseModel):
    techsummary: str

class FundamentalAnalyisReport(BaseModel):
    summary: str

class FinancialReport(BaseModel):
    report: str

@CrewBase
class FinancialAdvisor:

    def __init__(self, agents_config, tasks_config, stock_symbol):
        self.agents_config = agents_config
        self.tasks_config = tasks_config
        self.stock_symbol = stock_symbol
        # Use the global llm instance
        self.llm = llm
        self.serper_tool = SerperDevTool()
        self.reddit_tool = RedditSentimentAnalysisTool()
        # Use our custom Yahoo Finance News Tool
        self.yf_news_tool = YahooFinanceNewsTool()
        self.yf_tech_tool = YFinanceTechnicalAnalysisTool()
        self.yf_fundamental_tool = YFinanceFundamentalAnalysisTool()

    @agent
    def researcher(self) -> Agent:
        print(f"Stock Symbol Passed to Agent: {self.stock_symbol}")
        return Agent(
            role=self.agents_config['researcher']['role'],
            goal=self.agents_config['researcher']['goal'].format(stock_symbol=self.stock_symbol),
            backstory=self.agents_config['researcher']['backstory'],
            verbose=True,
            allow_delegation=False,
            tools=[self.serper_tool, self.yf_news_tool],
            llm=self.llm
        )

    @task
    def research_task(self) -> Task:
        print(f"Stock Symbol Passed to Task: {self.stock_symbol}")
        return Task(
            description=self.tasks_config['research_task']['description'].format(stock_symbol=self.stock_symbol),
            expected_output=self.tasks_config['research_task']['expected_output'],
            agent=self.researcher(),
            output_json=ResearchReport
        )
    
    @agent
    def technical_analyst(self)-> Agent:
        return Agent( 
            role=self.agents_config['technical_analyst']['role'],
            goal=self.agents_config['technical_analyst']['goal'].format(stock_symbol=self.stock_symbol),
            backstory=self.agents_config['technical_analyst']['backstory'],
            verbose=True,
            allow_delegation=False,
            tools=[self.yf_tech_tool],
            llm=self.llm
        )
    
    @task
    def technical_analysis_task(self) -> Task:
        print(f"Stock Symbol Passed to Task: {self.stock_symbol}")
        return Task(
            description=self.tasks_config['technical_analysis_task']['description'].format(stock_symbol=self.stock_symbol),
            expected_output=self.tasks_config['technical_analysis_task']['expected_output'],
            agent=self.technical_analyst(),
            output_json=TechnicalAnalysisReport
        )
    
    @agent
    def fundamental_analyst(self)-> Agent:
        return Agent( 
            role=self.agents_config['fundamental_analyst']['role'],
            goal=self.agents_config['fundamental_analyst']['goal'].format(stock_symbol=self.stock_symbol),
            backstory=self.agents_config['fundamental_analyst']['backstory'],
            verbose=True,
            allow_delegation=False,
            tools=[self.yf_fundamental_tool],
            llm=self.llm
        )

    @task
    def fundamental_analysis_task(self) -> Task:
        print(f"Stock Symbol Passed to Task: {self.stock_symbol}")
        return Task(
            description=self.tasks_config['fundamental_analysis_task']['description'].format(stock_symbol=self.stock_symbol),
            expected_output=self.tasks_config['fundamental_analysis_task']['expected_output'],
            agent=self.fundamental_analyst(),
            output_json=FundamentalAnalyisReport
        )
    
    @agent
    def reporter(self)-> Agent:
        return Agent( 
            role=self.agents_config['reporter']['role'],
            goal=self.agents_config['reporter']['goal'].format(stock_symbol=self.stock_symbol),
            backstory=self.agents_config['reporter']['backstory'],
            verbose=True,
            allow_delegation=False,
            tools=[self.reddit_tool, self.serper_tool, self.yf_fundamental_tool, self.yf_tech_tool, self.yf_news_tool],
            llm=self.llm
        )
    
    @task
    def report_task(self) -> Task:
        print(f"Stock Symbol Passed to Task: {self.stock_symbol}")
        return Task(
            description=self.tasks_config['report_task']['description'].format(stock_symbol=self.stock_symbol),
            expected_output=self.tasks_config['report_task']['expected_output'],
            agent=self.reporter(),
            output_json=FinancialReport
        )
    
    @crew
    def create_crew(self) -> Crew:
        """Creates the Stock Analysis Crew"""
        agents = [
            self.researcher(),
            self.technical_analyst(),
            self.fundamental_analyst(),
            self.reporter()
        ]
        tasks = [
            self.research_task(),
            self.technical_analysis_task(),
            self.fundamental_analysis_task(),
            self.report_task()
        ]
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )


def run_analysis(stock_symbol):
    # Instantiate the FinancialAdvisor class
    print(f"Running analysis for stock: {stock_symbol}")
    advisor = FinancialAdvisor(
        agents_config=agents_config,
        tasks_config=tasks_config,
        stock_symbol=stock_symbol
    )
    
    crew = advisor.create_crew()
    print("Crew created successfully")
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    analysis_result = run_analysis('AAPL')
    print("RESULT:", analysis_result)