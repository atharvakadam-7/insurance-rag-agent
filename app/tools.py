from langchain_core.tools import tool

from .rag import get_retriever, format_docs


@tool
def search_policy_docs(query: str) -> str:
    """Search the insurance policy documents for information relevant to the
    query. Use this whenever the user asks about coverage, exclusions,
    waiting periods, or any specific clause in a policy. Always call this
    before answering a coverage question — never answer from memory."""
    docs = get_retriever().invoke(query)
    if not docs:
        return "No relevant policy sections found for that query."
    return format_docs(docs)


@tool
def calculate_claim_reimbursement(
    claim_amount: float, coverage_percent: float, deductible: float = 0.0
) -> str:
    """Calculate the estimated reimbursement for a claim.
    claim_amount: the total claim value in rupees.
    coverage_percent: the coverage percentage from the policy, e.g. 80 for 80%.
      Retrieve this from the policy docs first — never guess a percentage.
    deductible: any deductible/excess that applies before coverage kicks in.
    """
    payable_base = max(claim_amount - deductible, 0)
    reimbursement = payable_base * (coverage_percent / 100)
    return (
        f"Claim amount: Rs {claim_amount:,.2f}\n"
        f"Deductible: Rs {deductible:,.2f}\n"
        f"Coverage: {coverage_percent}%\n"
        f"Estimated reimbursement: Rs {reimbursement:,.2f}"
    )


@tool
def compare_policy_clauses(clause_a: str, clause_b: str) -> str:
    """Compare two policy clause texts (already retrieved via
    search_policy_docs) on coverage, exclusions, and conditions. Pass the
    actual clause text, not a policy name — this tool doesn't retrieve
    anything itself."""
    return (
        "Compare these two policy clauses on coverage, exclusions, "
        f"and conditions:\n\nClause A:\n{clause_a}\n\nClause B:\n{clause_b}"
    )


TOOLS = [search_policy_docs, calculate_claim_reimbursement, compare_policy_clauses]
