import os
import json
import re
from difflib import get_close_matches
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings# Optimized for API
from langchain_core.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

load_dotenv()

# --- 1. EMBEDDING CONFIGURATION (STAYING LIGHT) ---
# This uses the Hugging Face API so you don't need torch or sentence-transformers
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# --- 2. VECTOR DATABASE LOADING ---
# Ensure these folders exist in your deployment repo
enterprise_db = Chroma(
    persist_directory="./vector_stores/enterprise", 
    embedding_function=embeddings, 
    collection_name="enterprise"
)

troubleshoot_db = Chroma(
    persist_directory="./vector_stores/troubleshoot", 
    embedding_function=embeddings, 
    collection_name="troubleshoot"
)

# --- 3. TICKET GENERATION LOGIC ---
issue_ticket_map = {
    "wifi not working": "TICKET-WIFI-001",
    "email issue": "TICKET-EMAIL-002",
    "vpn connection problem": "TICKET-VPN-003",
}

def _normalize(text: str) -> str:
    t = text.lower()
    t = t.replace("wi-fi", "wifi").replace("wi fi", "wifi")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def match_issue_to_ticket(summary: str) -> str:
    normalized = _normalize(summary)
    for key, tid in issue_ticket_map.items():
        if _normalize(key) in normalized:
            return tid
    return "HCL-DEFAULT"

@tool
def create_support_ticket(issue_summary: str, priority: str = "Medium"):
    """
    Final escalation tool. 
    Use ONLY when troubleshooting fails or user explicitly requests a ticket.
    """
    ticket_id = match_issue_to_ticket(issue_summary)
    ticket_data = {
        "ticket_id": ticket_id,
        "issue_summary": issue_summary,
        "priority": priority,
        "status": "Open"
    }
    
    # In production, you might write to a cloud DB instead of a local JSON
    with open("ticket.json", "w") as f:
        json.dump(ticket_data, f, indent=4)
        
    return f"Success: Ticket {ticket_id} created for: {issue_summary}"

# --- 4. RETRIEVER TOOLS ---
enterprise_tool = create_retriever_tool(
    enterprise_db.as_retriever(search_kwargs={"k": 5}),
    "hcl_strategic_intel",
    "Search for HCLTech's official business strategy, AI Labs, and partnership data."
)

troubleshoot_tool = create_retriever_tool(
    troubleshoot_db.as_retriever(search_kwargs={"k": 3}),
    "it_operational_intel",
    "MANDATORY first step for technical issues. Contains fixes for VPN, email, and WiFi."
)

tools = [enterprise_tool, troubleshoot_tool, create_support_ticket]
