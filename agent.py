import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from tools import tools

load_dotenv()

def get_agent_executor():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
    )

    system_prompt = (
        "You are an expert HCLTech IT Support Agent. Your goal is to assist users with technical issues efficiently and professionally.\n"
        "1. **Content First**: When answering questions based on knowledge base (content.pdf), you MUST traverse and refer to the context fully.\n"
        "2. **Natural Language**: Provide responses in clear, helpful natural language (NOT JSON) in the chat.\n"
        "3. **Troubleshooting**: If technical issues arise, guide the user step-by-step.\n"
        "4. **Ticket Generation**: If troubleshooting fails or the user requests a ticket, you MUST use the `create_support_ticket` tool. "
        "This tool will generate a 'ticket.json' file. Confirm to the user when this is done.\n"
        "5. **Format**: Use clear markdown in your chat responses (lists, bold text) for readability.\n"
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    
    return agent
