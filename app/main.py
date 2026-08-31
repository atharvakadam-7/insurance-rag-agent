from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .agent import build_agent

app = FastAPI(title="Insurance Policy Agent")

# Built once at startup, not per-request — rebuilding the agent (and its LLM
# client) on every call would add latency for no reason.
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    agent = get_agent()

    try:
        # Pass a proper HumanMessage object to the agent
        result = agent.invoke({"messages": [HumanMessage(content=req.question)]})

        # Extract the final AI message response
        final_message = result["messages"][-1]

        # Handle string or list-content outputs from modern ChatModels
        if isinstance(final_message.content, list):
            answer = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in final_message.content
            )
        else:
            answer = str(final_message.content)

        return QueryResponse(answer=answer)

    except Exception as e:
        # Prevent 500 crashes from bubbling up unhandled; return a clear
        # execution error instead
        raise HTTPException(status_code=500, detail=str(e))