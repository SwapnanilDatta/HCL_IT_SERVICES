import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from agent import get_agent_executor

app = FastAPI(title="HCLTech IT Support Agent API")

# Initialize the agent once on startup
agent = get_agent_executor()

# Define input schema
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    thread_id: Optional[str] = "default-thread"

# Define output schema
class ChatResponse(BaseModel):
    assistant_msg: str
    thread_id: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Prepare conversation history for the agent
        # LangGraph ReAct agent expects a dict with "messages"
        conversation = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Invoke the agent
        # config is used for thread persistence if you add a checkpointer later
        config = {"configurable": {"thread_id": request.thread_id}}
        response = agent.invoke({"messages": conversation}, config=config)

        # Extract the last message content
        assistant_msg = response["messages"][-1].content
        
        return ChatResponse(
            assistant_msg=assistant_msg,
            thread_id=request.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))