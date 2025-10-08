# Truly Agentic GCP Cost Optimizer - Agents make decisions and collaborate

import pandas as pd
import logging
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("c://code//agentic_ai//4_crewai//gcp_cost_optimizer.log"),
        logging.StreamHandler()
    ]
)

load_dotenv(override=True)

@tool
def get_gcp_billing_data(query_type: str) -> str:
    """Get GCP billing data based on agent's specific query"""
    logging.info(f"Tool: get_gcp_billing_data called with query_type='{query_type}'")
    try:
        df = pd.read_csv('c://code//agentic_ai//5_crew//gcp_billing.csv')
        
        summary = f"REAL DATA - {len(df)} records, Total cost: ₹{df['Total Cost (INR)'].sum():,.0f}\n"
        summary += f"Date range: {df['Usage Start Date'].min()} to {df['Usage End Date'].max()}\n"
        summary += f"Services: {df['Service Name'].unique()[:10].tolist()}\n"
        summary += f"Resource IDs: {df['Resource ID'].unique()[:10].tolist()}\n"
        summary += f"Regions: {df['Region/Zone'].unique()[:5].tolist()}\n"
        
        logging.info("Tool: get_gcp_billing_data completed successfully")
        return summary
    except Exception as e:
        logging.error(f"Tool: get_gcp_billing_data encountered error: {e}")
        return f"Error loading data: {str(e)}"

@tool
def query_resource_metrics(service_filter: str, threshold: int) -> str:
    """Query specific resource metrics with REAL resource details"""
    logging.info(f"Tool: query_resource_metrics called with service_filter='{service_filter}', threshold={threshold}")
    try:
        df = pd.read_csv('c://code//agentic_ai//5_crew//gcp_billing.csv')
        
        if 'compute' in service_filter.lower():
            filtered = df[df['Service Name'].str.contains('Compute|VM|Instance', case=False, na=False)]
            if len(filtered) == 0:
                logging.info("No compute resources found in the data")
                return "No compute resources found in the data"
            
            result = f"ACTUAL COMPUTE RESOURCES ({len(filtered)} found):\n"
            for _, row in filtered.head(5).iterrows():
                result += f"- {row['Resource ID']}: {row['Service Name']}, CPU: {row['CPU Utilization (%)']}%, Cost: ₹{row['Total Cost (INR)']:.0f}, Region: {row['Region/Zone']}\n"
            
            if threshold:
                underutilized = filtered[filtered['CPU Utilization (%)'] < threshold]
                result += f"\nUNDERUTILIZED (CPU < {threshold}%):\n"
                for _, row in underutilized.iterrows():
                    result += f"- {row['Resource ID']}: {row['CPU Utilization (%)']}% CPU, ₹{row['Total Cost (INR)']:.0f} cost\n"
                    
        elif 'storage' in service_filter.lower():
            filtered = df[df['Service Name'].str.contains('Storage|SQL|BigQuery|Disk', case=False, na=False)]
            result = f"ACTUAL STORAGE RESOURCES ({len(filtered)} found):\n"
            for _, row in filtered.head(5).iterrows():
                result += f"- {row['Resource ID']}: {row['Service Name']}, Cost: ₹{row['Total Cost (INR)']:.0f}\n"
        else:
            result = f"SAMPLE OF ALL RESOURCES:\n"
            for _, row in df.head(10).iterrows():
                result += f"- {row['Resource ID']}: {row['Service Name']}, Cost: ₹{row['Total Cost (INR)']:.0f}\n"
        
        logging.info("Tool: query_resource_metrics completed successfully")
        return result
    except Exception as e:
        logging.error(f"Tool: query_resource_metrics encountered error: {e}")
        return f"Query error: {str(e)}"

# --- Define Agents with Logging Observability ---
compute_analyst = Agent(
    role="Senior Compute Performance Analyst",
    goal="Identify compute inefficiencies and recommend specific rightsizing actions",
    backstory="""You are a seasoned cloud architect with 8 years at Google Cloud. 
    You're obsessed with compute efficiency and hate seeing wasted CPU cycles. 
    You always dig deep into utilization patterns before making recommendations.
    You prefer conservative estimates and always consider business impact.""",
    tools=[get_gcp_billing_data, query_resource_metrics],
    verbose=True,
    allow_delegation=True
)

financial_advisor = Agent(
    role="Cloud Financial Controller", 
    goal="Maximize cost savings while minimizing business risk",
    backstory="""You're the CFO's trusted advisor for cloud spending. 
    You've saved companies millions through smart optimization. 
    You're skeptical of aggressive changes and always want ROI analysis.
    You think in terms of monthly/quarterly budgets and business impact.""",
    tools=[get_gcp_billing_data, query_resource_metrics],
    verbose=True,
    allow_delegation=True
)

implementation_manager = Agent(
    role="Cloud Operations Manager",
    goal="Create actionable implementation plans that won't break production",
    backstory="""You've been burned by optimization changes that caused outages. 
    You're paranoid about production stability but recognize the need for efficiency.
    You always create staged rollout plans and require rollback procedures.
    You coordinate between teams and manage change timelines.""",
    tools=[get_gcp_billing_data, query_resource_metrics],
    verbose=True,
    allow_delegation=False
)

# --- Define Tasks with Logging ---
discovery_task = Task(
    description="""Analyze the GCP billing data to understand our compute landscape.
    
    YOU DECIDE what constitutes 'underutilized' based on your expertise.
    YOU CHOOSE which metrics matter most for our optimization.
    
    Consider:
    - What CPU utilization threshold should we use?
    - Are there seasonal patterns to consider?
    - Which instance types are most problematic?
    
    Make your own professional judgment calls.""",
    expected_output="Your expert analysis of compute utilization with YOUR recommended thresholds and priorities",
    agent=compute_analyst
)

financial_analysis_task = Task(
    description="""Based on the compute analysis, YOU DECIDE the business case for optimization.
    
    YOU DETERMINE:
    - What level of savings justifies the effort?
    - Which optimizations have acceptable risk?
    - What's the ROI timeline we should target?
    
    Challenge the compute analyst's recommendations if needed.
    YOU make the final call on financial priorities.""",
    expected_output="Your business case analysis with financial recommendations and risk assessment",
    agent=financial_advisor,
    context=[discovery_task]
)

implementation_plan_task = Task(
    description="""Create YOUR implementation strategy based on both technical and financial analysis.
    
    YOU DECIDE:
    - Which changes to implement first, second, third?
    - What's a safe rollout timeline?
    - How do we minimize production risk?
    - What rollback procedures do we need?
    
    You have the authority to reject recommendations that seem too risky.
    YOU create the final action plan.""",
    expected_output="Your detailed implementation roadmap with timelines, priorities, and risk mitigation",
    agent=implementation_manager,
    context=[discovery_task, financial_analysis_task]
)

# --- Define Crew with Logging ---
crew = Crew(
    agents=[compute_analyst, financial_advisor, implementation_manager],
    tasks=[discovery_task, financial_analysis_task, implementation_plan_task],
    process=Process.sequential,
    verbose=True,
    memory=True
)

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting AGENTIC GCP cost optimization process")
    print("Starting AGENTIC GCP cost optimization...")
    print("Agents will analyze, debate, and decide autonomously...")
    print("="*60)
    
    try:
        result = crew.kickoff()
        logging.info("Crew execution completed successfully")
        
        print("\nAGENTIC ANALYSIS COMPLETE:")
        print("="*60)
        print("This is what the agents DECIDED (not just calculated):")
        print(result)
        
        logging.info(f"Final decision:\n{result}")
    except Exception as e:
        logging.error(f"Crew execution failed: {e}")
        print("An error occurred during execution. Check logs for details.")
