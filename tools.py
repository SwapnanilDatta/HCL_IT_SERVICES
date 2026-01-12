from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools.retriever import create_retriever_tool

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load existing stores
enterprise_db = Chroma(persist_directory="./vector_stores/enterprise", embedding_function=embeddings, collection_name="enterprise")
troubleshoot_db = Chroma(persist_directory="./vector_stores/troubleshoot", embedding_function=embeddings, collection_name="troubleshoot")

from langchain.tools import tool
import json
import re
from difflib import get_close_matches

# Dictionary mapping technical issues to fixed ticket IDs
issue_ticket_map = {
    "wifi not working": "TICKET-WIFI-001",
    "email issue": "TICKET-EMAIL-002",
    "vpn connection problem": "TICKET-VPN-003",
    # Add more issue-to-ticket mappings as needed
}

# Helper functions for robust issue matching

def _normalize(text: str) -> str:
    t = text.lower()
    t = t.replace("wi-fi", "wifi").replace("wi fi", "wifi")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def match_issue_to_ticket(summary: str) -> str:
    normalized = _normalize(summary)

    # 1) Direct equality against normalized keys
    for key, tid in issue_ticket_map.items():
        if _normalize(key) == normalized:
            return tid

    # 2) Substring containment (key phrase within the user text)
    for key, tid in issue_ticket_map.items():
        if _normalize(key) in normalized:
            return tid

    # 3) Token overlap score
    norm_keys = {key: set(_normalize(key).split()) for key in issue_ticket_map}
    tokens = set(normalized.split())
    best = None
    best_score = 0.0
    for key, toks in norm_keys.items():
        inter = len(tokens & toks)
        score = inter / max(1, len(toks))
        if score > best_score:
            best_score = score
            best = key
    if best and best_score >= 0.6:
        return issue_ticket_map[best]

    # 4) Fuzzy match as last resort
    candidates = list(norm_keys.keys())
    matches = get_close_matches(normalized, [_normalize(k) for k in candidates], n=1, cutoff=0.8)
    if matches:
        for orig_key in candidates:
            if _normalize(orig_key) == matches[0]:
                return issue_ticket_map[orig_key]

    return "HCL-DEFAULT"

@tool
def create_support_ticket(issue_summary: str, priority: str = "Medium"):
    """
    Final escalation tool.
    Use when user explicitly requests ticket OR troubleshooting failed.
    """
    
    ticket_id = match_issue_to_ticket(issue_summary)
    
    ticket_data = {
        "ticket_id": ticket_id,
        "issue_summary": issue_summary,
        "priority": priority,
        "status": "Open"
    }
    
    with open("ticket.json", "w") as f:
        json.dump(ticket_data, f, indent=4)
        
    return (
        f"Success: Ticket {ticket_id} has been generated and saved to ticket.json.\n"
        f"Summary: {issue_summary}"
    )



enterprise_tool = create_retriever_tool(
    enterprise_db.as_retriever(search_kwargs={"k": 20}), # Get top 20 chunks to "traverse" more content
    "hcl_strategic_intel",
    "Query this for HCLTech's official strategy, AI Labs, and partnership data. Mandatory for corporate questions."
)
troubleshoot_tool = create_retriever_tool(
    troubleshoot_db.as_retriever(search_kwargs={"k": 5}),
    "it_operational_intel",
    "MANDATORY first step for all technical issues. Contains step-by-step fixes for email, VPN, and software."
)
tools = [enterprise_tool, troubleshoot_tool, create_support_ticket]

