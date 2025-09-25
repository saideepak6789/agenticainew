from openai import OpenAI
from dotenv import load_dotenv
import json

from agents import Agent, Runner, FunctionTool

load_dotenv(override=True)
client = OpenAI()

knowledge_base = {
    "shipping_time": "Our standard shipping time is 3-5 business days.",
    "return_policy": "You can return any product within 30 days of delivery.",
    "warranty": "All products come with a one-year warranty covering manufacturing defects.",
    "payment_methods": "We accept credit cards, debit cards, and PayPal.",
    "customer_support": "You can reach our support team 24/7 via email or chat."
}

# --- Tool callback function ---
async def faq_invoker(tool_context, params):
    # If params = '{"name": "Alice", "age": 30}'  # string
    #  then parse into a dict
    #  params = {"name": "Bob", "age": 25}  # dict
    # Else directly use params as it is already a JSON, i.e. a dict
    args = json.loads(params) if isinstance(params, str) else params
    user_query = args.get("topic", "").lower()

    # Try to find a matching topic
    for topic_key, answer in knowledge_base.items():
        if topic_key in user_query:
            return answer
        
    return "I'm sorry, but I couldn't find specific information about that topic. Please check the company's website or contact customer support directly for detailed assistance."

# --- Tool schema ---
faq_schema = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "The topic or question asked by the customer."
        }
    },
    "required": ["topic"]
}

# --- Register FunctionTool ---
faq_tool = FunctionTool(
    name="get_faq_answer",
    description="Provides answers to frequently asked customer support questions.",
    params_json_schema=faq_schema,
    on_invoke_tool=faq_invoker
)

# --- Define Agent ---
faq_agent = Agent(
    name="Customer Support Bot",
    instructions="You are a helpful customer support assistant. "
                 # "Answer questions based on the knowledge base or use the provided FAQ tool.",
                 "Answer questions strictly with the help of the provided FAQ tool.",
    tools=[faq_tool]
)

# --- Chat function ---
# async makes it a coroutine
# Calls Runner.run, which is also async, meaning it might call openai.chat.completions.create
#  await is used to wait for the result of the Runner.run call
#  When done, it extracts session.final_output and returns it
async def chat_with_support(message):
    session = await Runner.run(faq_agent, message)
    return session.final_output

# --- Main loop for testing ---
if __name__ == "__main__":
    import asyncio

    async def main():
        print("Customer Support Bot is running. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting.")
                break
            # Calls chat_with_support and as we saw, pauses till
            #  Runner.run completes
            response = await chat_with_support(user_input)
            print("Bot:", response)

    asyncio.run(main())
