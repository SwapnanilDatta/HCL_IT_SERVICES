from agent import get_agent_executor
from dotenv import load_dotenv
import os
import json

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def start_chat():
    agent = get_agent_executor()
    print("🚀 HCL Agent is Live. Type 'exit' to quit.\n")

    conversation = []
    # Issue context can be tracked in the conversation history or state if needed, 
    # but the agent is now stateless per turn in this simple loop, relying on conversation history.
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        conversation.append({"role": "user", "content": user_input})
        
        try:
            response = agent.invoke({
                "messages": conversation
            })

            assistant_msg = response["messages"][-1].content
            print(f"\nAgent: {assistant_msg}\n")

            conversation.append({"role": "assistant", "content": assistant_msg})
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    start_chat()
