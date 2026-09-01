from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from .agent import build_agent

app = FastAPI(title="Insurance Policy Agent")

_agent = None

NOT_FOUND_ANSWER = "The provided policy documents do not contain information about this."


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
        result = agent.invoke(
            {"messages": [HumanMessage(content=req.question)]},
            config={"recursion_limit": 15},
        )
        final_message = result["messages"][-1]

        if isinstance(final_message.content, list):
            answer = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in final_message.content
            )
        else:
            answer = str(final_message.content)

        # Some runs end on a tool-call message with no text content instead
        # of a proper final answer. Treat that the same as "not found"
        # rather than returning an empty string to the user.
        if not answer.strip():
            answer = NOT_FOUND_ANSWER

        return QueryResponse(answer=answer)

    except GraphRecursionError:
        # The agent kept searching without converging on an answer. This is
        # functionally the same as "not found in the docs" from the user's
        # perspective, so respond cleanly instead of a 500.
        return QueryResponse(answer=NOT_FOUND_ANSWER)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))