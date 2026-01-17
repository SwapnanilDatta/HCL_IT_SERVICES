import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
from agent import get_agent_executor

app = FastAPI(title="HCLTech IT Support Agent API")

# Configure CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent once on startup
agent = get_agent_executor()

# Define input schema
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    thread_id: Optional[str] = "default-thread"

# Define output schemas
class ChatResponse(BaseModel):
    assistant_msg: str
    thread_id: str
    mode: str = "chat"

class TicketResponse(BaseModel):
    mode: str = "ticket"
    ticket_id: str
    issue_summary: str
    priority: str
    status: str
    thread_id: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def _parse_ticket_response(response_text: str) -> Optional[dict]:
    """Extract ticket data from agent response if ticket was created"""
    # Check for ticket creation indicators
    ticket_indicators = ["Success: Ticket", "ticket has been generated", "ticket number is", "ticket created"]
    is_ticket_response = any(indicator.lower() in response_text.lower() for indicator in ticket_indicators)
    
    if not is_ticket_response:
        return None
    
    try:
        if os.path.exists("ticket.json"):
            with open("ticket.json", "r") as f:
                ticket_data = json.load(f)
            # Delete the file after reading to prevent reuse in future requests
            os.remove("ticket.json")
            return ticket_data
    except Exception:
        pass
    return None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Prepare conversation history for the agent
        conversation = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Invoke the agent
        config = {"configurable": {"thread_id": request.thread_id}}
        response = agent.invoke({"messages": conversation}, config=config)

        # Extract the last message content
        assistant_msg = response["messages"][-1].content
        
        # Check if a ticket was generated in this interaction
        ticket_data = _parse_ticket_response(assistant_msg)
        
        if ticket_data:
            # Return ticket generation response
            return {
                "mode": "ticket",
                "ticket_id": ticket_data.get("ticket_id"),
                "issue_summary": ticket_data.get("issue_summary"),
                "priority": ticket_data.get("priority"),
                "status": ticket_data.get("status"),
                "thread_id": request.thread_id,
                "message": assistant_msg
            }
        else:
            # Return normal chat response
            return {
                "mode": "chat",
                "assistant_msg": assistant_msg,
                "thread_id": request.thread_id
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
