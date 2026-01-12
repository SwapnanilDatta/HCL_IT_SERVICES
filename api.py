import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from agent import get_agent_executor

app = FastAPI(title="HCLTech IT Support Agent API")

# Initialize the agent once on startup
agent = get_agent_executor()

# Define input/output schemas
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    thread_id: Optional[str] = "default-thread"

class ChatResponse(BaseModel):
    assistant_msg: str
    thread_id: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        conversation = [{"role": m.role, "content": m.content} for m in request.messages]
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Invoke the agent
        response = agent.invoke({"messages": conversation}, config=config)
        assistant_msg = response["messages"][-1].content
        
        return ChatResponse(
            assistant_msg=assistant_msg,
            thread_id=request.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CRITICAL: Programmatic binding for Render
if __name__ == "__main__":
    # Render provides the port in the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    # Must bind to 0.0.0.0 for external visibility
    uvicorn.run("api:app", host="0.0.0.0", port=port)
