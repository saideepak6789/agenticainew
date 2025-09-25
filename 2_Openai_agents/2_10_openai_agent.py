import pandas as pd
from collections import OrderedDict
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool
import asyncio

load_dotenv(override=True)

# ---------- Config ----------
DB_PATH = "conversations.db"
CSV_PATH = "c:\\code\\agenticai\\2_openai_agents\\Conversation.csv"
MODEL = "gpt-4o-mini"
client = OpenAI()

# ---------- LRU Short-Term Memory ----------
class LRUCache:
    def __init__(self, max_size=500):
        # OrderedDict is a dictionary that remembers the order of insertion
        # Helps implement LRU
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key): # get the value for a key
        # print("In get function ", key)
        if key in self.cache:
            self.cache.move_to_end(key) # move the key to the end, since it was recently used
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False) # remove the least recently used item
            
    def contains(self, key): # check if the key is in the cache
        return key in self.cache

short_term_memory = LRUCache(max_size=500)

# ---------- DB Setup ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS faqs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL UNIQUE,
        answer TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def load_csv_into_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM faqs")
    rec_count = cursor.fetchone()[0]
    if rec_count > 0:
        print("Number of records in DB: ", rec_count)
        conn.close()
        return

    df = pd.read_csv(CSV_PATH)
    print("Loading CSV into DB and preloading short-term memory...")

    for i, (_, row) in enumerate(df.iterrows()):
        topic = row["question"].strip()
        answer = row["answer"].strip()

        # Insert into long-term DB
        try:
            cursor.execute(
                "INSERT INTO faqs (topic, answer) VALUES (?, ?)",
                (topic, answer)
            )
        except sqlite3.IntegrityError:
            continue

        # Preload short-term memory with first 500 entries
        if i < 500:
            short_term_memory.put(topic, answer)

    conn.commit()
    conn.close()
    print(f"Loaded {len(df)} entries into long-term memory.")
    print(f"Preloaded {min(len(df), 500)} entries into short-term memory.")


# ---------- FunctionTool for DB ----------
@function_tool
def query_faq_db(topic: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # print ("in query_faq_db ", topic)
    cursor.execute("SELECT answer FROM faqs WHERE topic = ?", (topic,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "Sorry, I cannot help you."

# ---------- FunctionTool for Short-Term Memory ----------
@function_tool
def query_short_term_memory(topic: str) -> str:
    result = short_term_memory.get(topic)
    return result if result else "Sorry, I cannot help you."

# ---------- Create Agent ----------
agent = Agent(
    model=MODEL,
    tools=[query_faq_db, query_short_term_memory],
    name="CustomerServiceAgent",
    instructions=(
        "You are a customer service agent. "
        "You can only answer using short-term or long-term memory via the provided tools. "
        "If the answer is not found, respond exactly with: 'Sorry, I cannot help you.'"
    )
)

# runner = Runner.run(agent)

# ---------- Async Wrapper ----------
async def get_agent_response(user_query: str) -> str:
    user_query = user_query.strip()
    # print(user_query)

    # Check short-term memory first
    if short_term_memory.get(user_query):
        # print("Found in STM")
        return f"(short-term memory) {short_term_memory.get(user_query)}"

    # Add empty placeholder
    if not short_term_memory.contains(user_query):
        # print("Not found in STM")
        short_term_memory.put(user_query, "")

    # Run agent
    response = await Runner.run(agent, user_query)
    answer = response.final_output.strip()
    
    # print("Answer:", answer)

    # Update short-term memory if an actual answer is returned
    if answer != "Sorry, I cannot help you.":
        short_term_memory.put(user_query, answer)

    return f"(AI) {answer}"

# ---------- Run ----------
async def main():
    init_db()
    load_csv_into_db()
    print("Agent Ready! Short + Long Term Memory. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = await get_agent_response(user_input)
        print("Bot:", answer)

if __name__ == "__main__":
    asyncio.run(main())