import gradio as gr
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import os

load_dotenv(override=True)

# Agents
news_agent = Agent(
    role="Market News Analyst",
    goal="Analyze stock from market news and sentiment perspective",
    backstory="Expert in market sentiment analysis with deep understanding of news impact on stock prices",
    verbose=True
)

financial_agent = Agent(
    role="Financial Analyst", 
    goal="Analyze stock from financial metrics and fundamentals perspective",
    backstory="CFA with expertise in financial statement analysis and valuation models",
    verbose=True
)

sector_agent = Agent(
    role="Sector & Competition Analyst",
    goal="Analyze stock from competitive landscape and sector prospects perspective", 
    backstory="Industry expert specializing in competitive analysis and sector trends",
    verbose=True
)

review_agent = Agent(
    role="Senior Investment Analyst",
    goal="Critically review all analyses and provide comprehensive investment summary",
    backstory="Senior analyst with 15+ years experience in synthesizing complex investment research",
    verbose=True
)

def analyze_stock(stock_symbol, company_name=""):
    if not stock_symbol:
        return "Please enter a stock symbol"
    
    # Create company identifier
    company_id = f"{company_name} ({stock_symbol})" if company_name else stock_symbol
    
    # Tasks
    news_task = Task(
        description=f"Analyze {company_id} from market news and sentiment perspective. Research recent news, market sentiment, analyst ratings, and media coverage impact on stock performance.",
        expected_output="Market news analysis with sentiment assessment and news impact evaluation",
        agent=news_agent
    )
    
    financial_task = Task(
        description=f"Analyze {company_id} from financial fundamentals perspective. Evaluate key financial metrics, ratios, earnings trends, revenue growth, profitability, and valuation.",
        expected_output="Financial analysis with key metrics, ratios, and valuation assessment",
        agent=financial_agent
    )
    
    sector_task = Task(
        description=f"Analyze {company_id} from competitive and sector perspective. Research industry trends, competitive positioning, market share, and sector growth prospects.",
        expected_output="Competitive and sector analysis with industry positioning assessment",
        agent=sector_agent
    )
    
    review_task = Task(
        description="Critically review all previous analyses and synthesize findings into a comprehensive investment summary with clear recommendations.",
        expected_output="Critical review and investment recommendation with risk assessment",
        agent=review_agent,
        context=[news_task, financial_task, sector_task]
    )
    
    # Create and run crew
    crew = Crew(
        agents=[news_agent, financial_agent, sector_agent, review_agent],
        tasks=[news_task, financial_task, sector_task, review_task],
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Analysis failed: {str(e)}\n\nNote: This demo requires API keys for full functionality."

# Gradio Interface
with gr.Blocks(title="Stock Analysis Crew") as demo:
    gr.Markdown("#Stock Analysis Crew")
    gr.Markdown("Multi-agent stock analysis using CrewAI with market news, financial, and sector perspectives")
    
    with gr.Row():
        with gr.Column(scale=1):
            stock_input = gr.Textbox(
                label="Stock Symbol",
                placeholder="e.g., AAPL, TSLA, MSFT",
                value="AAPL"
            )
            company_input = gr.Textbox(
                label="Company Name (Optional)",
                placeholder="e.g., Apple Inc."
            )
            analyze_btn = gr.Button("Analyze Stock", variant="primary")
        
        with gr.Column(scale=2):
            output = gr.Textbox(
                label="Analysis Results",
                lines=20,
                max_lines=30,
                show_copy_button=True
            )
    
    analyze_btn.click(
        fn=analyze_stock,
        inputs=[stock_input, company_input],
        outputs=output
    )
    
    gr.Markdown("""
    ### Agent Roles:
    - **Market News Analyst**: Analyzes market sentiment and news impact
    - **Financial Analyst**: Evaluates financial metrics and fundamentals  
    - **Sector Analyst**: Assesses competitive landscape and industry trends
    - **Senior Analyst**: Reviews all inputs and provides investment summary
    """)

if __name__ == "__main__":
    demo.launch()