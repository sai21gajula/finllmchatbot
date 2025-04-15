from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from tools.yf_tech_analysis import YFinanceTechnicalAnalysisTool
from tools.yf_fundamental_analysis import YFinanceFundamentalAnalysisTool
from tools.sentiment_analysis import RedditSentimentAnalysisTool
from tools.yahoo_finance_tool import YahooFinanceNewsTool
from tools.AlphaVantage_finance_tool import AlphaVantageNewsTool
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import yaml
import time
import json

# Environment Variables
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["REDDIT_CLIENT_ID"] = os.getenv("REDDIT_CLIENT_ID")
os.environ["REDDIT_CLIENT_SECRET"] = os.getenv("REDDIT_CLIENT_SECRET")
os.environ["REDDIT_USER_AGENT"] = os.getenv("REDDIT_USER_AGENT")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["ALPHA_VANTAGE_API_KEY"] = os.getenv("ALPHA_VANTAGE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Multiple Groq accounts
groq_account1 = os.getenv("GROQ_API_KEY_1", os.getenv("GROQ_API_KEY"))
groq_account2 = os.getenv("GROQ_API_KEY_2", groq_account1)

print("API Keys loaded:")
print(f"Gemini API Key: {gemini_api_key[:5]}..." if gemini_api_key else "Gemini API Key: Not found")
print(f"Groq Account 1: {groq_account1[:5]}..." if groq_account1 else "Groq Account 1: Not found")
print(f"Groq Account 2: {groq_account2[:5]}..." if groq_account2 else "Groq Account 2: Not found")
print(f"Alpha Vantage API Key: {alpha_vantage_api_key[:5]}..." if alpha_vantage_api_key else "Alpha Vantage API Key: Not found")

# Initialize specialized LLMs
gemini_research_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_api_key,
    temperature=0.2,
)

groq_technical_llm = LLM(
    model="groq/llama3-8b-8192", 
    api_key=groq_account1,
    temperature=0.1,
)

groq_fundamental_llm = LLM(
    model="groq/llama3-8b-8192",
    api_key=groq_account2,
    temperature=0.2,
)

gemini_reporter_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_api_key,
    temperature=0.5,
)

# Fix the path to yaml files
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
        
        # Initialize tools
        self.serper_tool = SerperDevTool()
        self.reddit_tool = RedditSentimentAnalysisTool()
        
        # Initialize both news tools
        self.yf_news_tool = YahooFinanceNewsTool()
    
    # Initialize the Alpha Vantage tool with the API key
        if alpha_vantage_api_key:
            self.alpha_news_tool = AlphaVantageNewsTool(api_key=alpha_vantage_api_key)
            print(f"Alpha Vantage API key found, initialized tool")
        else:
        # Create a dummy tool that will return explanatory message
            print("No Alpha Vantage API key found in environment")
            self.alpha_news_tool = None

    @agent
    def researcher(self) -> Agent:
        print(f"Stock Symbol Passed to Agent: {self.stock_symbol}")
        
        # Determine which news tool to use as primary
        news_tools = []
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            print("Using Alpha Vantage as primary news source")
            news_tools = [self.alpha_news_tool]
       
        
        return Agent(
            role=self.agents_config['researcher']['role'],
            goal=self.agents_config['researcher']['goal'].format(stock_symbol=self.stock_symbol),
            backstory=self.agents_config['researcher']['backstory'],
            verbose=True,
            allow_delegation=False,
            tools=[self.serper_tool] + news_tools,
            llm=gemini_research_llm
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
    def reporter(self)-> Agent:
        # Determine which news tool to use as primary
        news_tools = []
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            news_tools = [self.alpha_news_tool]
        else:
            news_tools = [self.yf_news_tool]
            
        return Agent( 
            role=self.agents_config['reporter']['role'],
            goal=self.agents_config['reporter']['goal'].format(stock_symbol=self.stock_symbol),
            backstory=self.agents_config['reporter']['backstory'],
            verbose=True,
            allow_delegation=False,
            tools=[self.reddit_tool, self.serper_tool] + news_tools,
            llm=gemini_reporter_llm
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

    def execute_technical_analysis_directly(self):
        """
        Execute technical analysis directly without using an agent
        """
        print(f"Executing technical analysis directly for {self.stock_symbol}...")
        try:
            # Create and use the tool directly
            tech_tool = YFinanceTechnicalAnalysisTool()
            tech_result = tech_tool._run(ticker=self.stock_symbol, period="1y")
            
            # Process the result
            tech_summary = f"""
            Technical Analysis Report for {self.stock_symbol}:
            
            Current Price: ${tech_result.get('current_price', 'N/A')}
            
            Moving Averages:
            - 50-day SMA: ${tech_result.get('sma_50', 'N/A')}
            - 200-day SMA: ${tech_result.get('sma_200', 'N/A')}
            
            Momentum Indicators:
            - RSI: {tech_result.get('rsi', 'N/A')}
            - MACD: {tech_result.get('macd', 'N/A')}
            
            Support Levels: {', '.join([f"${level:.2f}" for level in tech_result.get('support_levels', [])])}
            
            Resistance Levels: {', '.join([f"${level:.2f}" for level in tech_result.get('resistance_levels', [])])}
            
            Chart Patterns: {', '.join(tech_result.get('identified_patterns', ['No significant patterns identified']))}
            
            Average True Range (ATR): {tech_result.get('atr', 'N/A')}
            
            Overall Technical Assessment: {
                "Bullish" if tech_result.get('sma_50', 0) > tech_result.get('sma_200', 0) and tech_result.get('rsi', 0) > 50 else
                "Bearish" if tech_result.get('sma_50', 0) < tech_result.get('sma_200', 0) and tech_result.get('rsi', 0) < 50 else
                "Neutral"
            }
            """
            
            class TechResult:
                def __init__(self, text):
                    self.techsummary = text
            
            return TechResult(tech_summary)
            
        except Exception as e:
            print(f"Error in technical analysis: {e}")
            error_message = f"Technical analysis could not be completed: {str(e)}"
            return TechResult(error_message)

    def execute_fundamental_analysis_directly(self):
        """
        Execute fundamental analysis directly without using an agent
        """
        print(f"Executing fundamental analysis directly for {self.stock_symbol}...")
        try:
            # Create and use the tool directly
            fund_tool = YFinanceFundamentalAnalysisTool()
            fund_result = fund_tool._run(ticker=self.stock_symbol)
            
            # Process the result
            fund_summary = f"""
            Fundamental Analysis Report for {self.stock_symbol}:
            
            Company: {fund_result.get('Company Name', 'N/A')}
            Sector: {fund_result.get('Sector', 'N/A')}
            Industry: {fund_result.get('Industry', 'N/A')}
            
            Key Ratios:
            - P/E Ratio: {fund_result.get('Key Ratios', {}).get('P/E Ratio', 'N/A')}
            - Forward P/E: {fund_result.get('Key Ratios', {}).get('Forward P/E', 'N/A')}
            - P/B Ratio: {fund_result.get('Key Ratios', {}).get('P/B Ratio', 'N/A')}
            - Debt to Equity: {fund_result.get('Key Ratios', {}).get('Debt to Equity', 'N/A')}
            - ROE: {fund_result.get('Key Ratios', {}).get('ROE', 'N/A')}
            
            Growth Rates:
            - Revenue Growth (YoY): {fund_result.get('Growth Rates', {}).get('Revenue Growth (YoY)', 'N/A')}
            - Net Income Growth (YoY): {fund_result.get('Growth Rates', {}).get('Net Income Growth (YoY)', 'N/A')}
            
            Valuation:
            - Market Cap: ${fund_result.get('Valuation Metrics', {}).get('Market Cap', 'N/A')}
            - Enterprise Value: ${fund_result.get('Valuation Metrics', {}).get('Enterprise Value', 'N/A')}
            - DCF Valuation: ${fund_result.get('Simple DCF Valuation', 'N/A')}
            
            Overall Fundamental Assessment: {
                "Strong Buy" if fund_result.get('Key Ratios', {}).get('ROE', 0) > 0.2 and
                              fund_result.get('Growth Rates', {}).get('Revenue Growth (YoY)', 0) > 0.1 else
                "Buy" if fund_result.get('Key Ratios', {}).get('ROE', 0) > 0.15 else
                "Hold" if fund_result.get('Key Ratios', {}).get('ROE', 0) > 0.1 else
                "Sell"
            }
            """
            
            class FundResult:
                def __init__(self, text):
                    self.summary = text
            
            return FundResult(fund_summary)
            
        except Exception as e:
            print(f"Error in fundamental analysis: {e}")
            error_message = f"Fundamental analysis could not be completed: {str(e)}"
            return FundResult(error_message)

    def run_sequential_analysis(self):
        """
        Run the tasks sequentially with error handling and delays between tasks.
        """
        print(f"Starting sequential analysis for {self.stock_symbol}")
    
        results = {}
    
        try:
            # 1. Run Research Task with Gemini
            print("Starting Research Task with Gemini...")
            research_agent = self.researcher()
            research_task = self.research_task()
            results['research'] = research_task.execute_sync(agent=research_agent)
            print("Research Task completed successfully!")
        
            # Save interim result
            self._save_interim_result('research', results['research'])
        
            # Wait before next task
            print("Waiting 45 seconds before Technical Analysis...")
        
            # 2. Execute Technical Analysis directly
            print(f"Executing technical analysis directly for {self.stock_symbol}...")
            try:
                tech_result = self.execute_technical_analysis_directly()
                results['technical'] = tech_result
                print("Technical Analysis completed directly!")
            
                # Save interim result
                self._save_interim_result('technical', results['technical'])
            except Exception as e:
                print(f"Error with technical analysis: {e}")
                class TechError:
                    def __init__(self, error):
                        self.techsummary = f"Technical analysis failed: {error}"
                results['technical'] = TechError(str(e))
        
            # Wait before next task
            print("Waiting 45 seconds before Fundamental Analysis...")
        
            # 3. Execute Fundamental Analysis directly
            print(f"Executing fundamental analysis directly for {self.stock_symbol}...")
            try:
                fund_result = self.execute_fundamental_analysis_directly()
                results['fundamental'] = fund_result
                print("Fundamental Analysis completed directly!")
            
                # Save interim result
                self._save_interim_result('fundamental', results['fundamental'])
            except Exception as e:
                print(f"Error with fundamental analysis: {e}")
                class FundError:
                    def __init__(self, error):
                        self.summary = f"Fundamental analysis failed: {error}"
                results['fundamental'] = FundError(str(e))
        
            # Wait before final task
            print("Waiting 45 seconds before Report Task...")
        
            # 4. Run Reporter Task with Gemini
            print("Starting Report Task with Gemini...")
            reporter_agent = self.reporter()
            report_task = self.report_task()

            # Clean function to handle text formatting issues
            def clean_text(text):
                if not text:
                    return "No data available"
                # Replace any unusual character sequences and clean up formatting
                text = text.replace("*∗∗", "**")
                text = text.replace("∗∗", "**")
                text = text.replace("−", "-")
                # Remove excessive spaces and format issues
                while "  " in text:
                    text = text.replace("  ", " ")
                # Fix newline issues
                text = text.replace("\n\n\n", "\n\n")
                return text

            # Extract and clean text from results
            research_text = results['research'].researchreport if hasattr(results['research'], 'researchreport') else 'Research data not available'
            technical_text = results['technical'].techsummary if hasattr(results['technical'], 'techsummary') else 'Technical analysis data not available'
            fundamental_text = results['fundamental'].summary if hasattr(results['fundamental'], 'summary') else 'Fundamental analysis data not available'

            research_text = clean_text(research_text)
            technical_text = clean_text(technical_text)
            fundamental_text = clean_text(fundamental_text)

            # Create a formatted markdown string
            context_str = f"""# Investment Analysis for {self.stock_symbol}

            ## Research Report
            {research_text}

            ## Technical Analysis   
            {technical_text}

            ## Fundamental Analysis
            {fundamental_text}

            Based on the above information, please create a comprehensive investment report with these sections:
            1. Executive Summary with investment recommendation
            2. Company Snapshot with key facts
            3. Financial Highlights including key metrics
            4. Technical Analysis summary
            5. Fundamental Analysis strengths and concerns
            6. Risk and Opportunity assessment 
            7. Investment Thesis with bull and bear cases
            8. 12-month Price Target forecast

            Format your response as clean, well-structured markdown.
            """

            # Use try/except to handle potential errors
            try:
                results['report'] = report_task.execute_sync(agent=reporter_agent, context=context_str)
                print("Report Task completed successfully!")
            except Exception as e:
                print(f"Error executing report task: {str(e)}")
                # Create a fallback report
                fallback_report = f"""# Investment Report for {self.stock_symbol}

                ## Executive Summary
                This is an automated fallback report generated due to an error in the AI reporting process.

                ## Research Summary
                {research_text}

                ## Technical Analysis   
                {technical_text}

                ## Fundamental Analysis
                {fundamental_text}

                Please review the above data to make an investment decision."""
                
                class ReportResult:
                    def __init__(self, report_text):
                        self.report = report_text
                
                results['report'] = ReportResult(fallback_report)
                print("Generated fallback report due to error.")

            # Save final result
            self._save_interim_result('report', results['report'])
            
            return results
        
        except Exception as e:
            print(f"Error in sequential analysis: {str(e)}")
            return {'error': str(e)}
    
    def _save_interim_result(self, task_name, result):
        """Save interim results to disk as they're completed."""
        os.makedirs(f'results/{self.stock_symbol}', exist_ok=True)
        
        # Convert to dict for JSON serialization
        if hasattr(result, "__dict__"):
            result_dict = result.__dict__
        else:
            result_dict = {"data": str(result)}
            
        # Save to JSON file
        with open(f'results/{self.stock_symbol}/{task_name}_{time.strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(result_dict, f, indent=4, default=str)
        
        print(f"Saved {task_name} result to disk")


def run_analysis(stock_symbol):
    # Instantiate the FinancialAdvisor class
    print(f"Running analysis for stock: {stock_symbol}")
    advisor = FinancialAdvisor(
        agents_config=agents_config,
        tasks_config=tasks_config,
        stock_symbol=stock_symbol
    )
    
    # Use the sequential execution method
    result = advisor.run_sequential_analysis()
    
    # Ensure 'report' is directly accessible
    if 'report' in result and hasattr(result['report'], 'report'):
        # Format the return value to make 'report' directly accessible
        formatted_result = {
            'report': result['report'].report,  # Extract the actual report text
            'research': result.get('research', {}),
            'technical': result.get('technical', {}),
            'fundamental': result.get('fundamental', {})
        }
        return formatted_result
    
    # If report isn't available, return what we have
    return result


if __name__ == "__main__":
    # Run analysis for a stock symbol
    stock_symbol = input("Enter stock symbol to analyze: ").upper()
    if not stock_symbol:
        stock_symbol = "AAPL"  # Default
    
    print(f"Starting analysis for {stock_symbol}...")
    analysis_result = run_analysis(stock_symbol)
    print("Analysis completed!")
    
    # Print summary of results
    print("\n=== ANALYSIS SUMMARY ===")
    for task_name in analysis_result:
        print(f"✅ {task_name.capitalize()} task completed")
    
    print("\nResults saved in the 'results' directory.")