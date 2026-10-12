"""Three typed tools. Owner: Person 3.

Each tool = Pydantic input schema + typed function. Bad input must be caught
and returned to the agent as feedback (one retry).

The graph should call tools through run_tool(), which validates the input and
turns any error into a plain-text message the agent can read and fix.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from src.settings import DOMAINS, TOP_K

# --- Reimbursement rules (from the Atlas finance policies) -------------------
# TODO(Person 4): move this block into src/settings.py (contract: no
# hard-coded numbers outside settings). Kept here until the team agrees.
# (item type, scope) -> (limit per unit, currency)
REIMBURSEMENT_CAPS = {
    ("hotel_night", "domestic"): (1500, "EGP"),
    ("hotel_night", "international"): (200, "USD"),
    ("taxi_leg", "domestic"): (800, "EGP"),
    ("taxi_leg", "international"): (800, "EGP"),
    ("client_dinner", "domestic"): (2000, "EGP"),
    ("client_dinner", "international"): (2000, "EGP"),
}
# scope -> (daily rate, currency). Per diem is a fixed rate, not a cap.
PER_DIEM_RATES = {"domestic": (500, "EGP"), "international": (75, "USD")}
# -----------------------------------------------------------------------------

_POLICY_ID_PREFIX = {"HR": "hr", "IT": "it", "FIN": "finance", "FINANCE": "finance"}
_SNIPPET_CHARS = 500


class PolicyLookupInput(BaseModel):
    """Input for looking up a policy by ID or topic."""

    query: str = Field(min_length=2)


class ReimbursementInput(BaseModel):
    """Input for the reimbursement calculator.

    Each item is a dict with:
      type: "hotel_night" | "taxi_leg" | "client_dinner" | "per_diem"
      scope: "domestic" | "international"
      quantity: nights / legs / people / days (default 1)
      amount: claimed amount PER UNIT (not needed for per_diem)
      currency: "EGP" | "USD" (not needed for per_diem)
    """

    items: list[dict] = Field(min_length=1)


class ListLeaveTypesInput(BaseModel):
    """Input for listing leave types (no arguments needed)."""


class _Item(BaseModel):
    """One validated reimbursement line."""

    type: Literal["hotel_night", "taxi_leg", "client_dinner", "per_diem"]
    scope: Literal["domestic", "international"]
    quantity: int = Field(default=1, ge=1)
    amount: float | None = Field(default=None, ge=0)
    currency: Literal["EGP", "USD"] | None = None


def _format_chunks(chunks) -> str:
    """Turn chunks into text the agent can read, with the citation tag on each."""
    if not chunks:
        return "No matching passages found."
    parts = []
    for c in chunks:
        snippet = c.text.strip().replace("\n", " ")[:_SNIPPET_CHARS]
        parts.append(f"[{c.source_filename} | {c.chunk_id}] {snippet}")
    return "\n\n".join(parts)


def _guess_domain(query: str) -> str | None:
    """Read the domain from a policy ID such as HR-VAC-001 or IT-002."""
    match = re.search(r"\b(HR|IT|FINANCE|FIN)[-_ ]?[A-Z0-9]", query.upper())
    return _POLICY_ID_PREFIX[match.group(1)] if match else None


def policy_lookup(args: PolicyLookupInput) -> str:
    """Look up a policy."""
    from src.retriever import retrieve  # lazy: needs the index

    domain = _guess_domain(args.query)
    if domain:
        chunks = retrieve(args.query, domain, TOP_K)
    else:  # no ID given: search every domain and keep the best matches
        pool = [c for d in DOMAINS for c in retrieve(args.query, d, TOP_K)]
        chunks = sorted(pool, key=lambda c: c.score, reverse=True)[:TOP_K]
    return _format_chunks(chunks)


def reimbursement_calculator(args: ReimbursementInput) -> str:
    """Calculate a reimbursement total."""
    lines: list[str] = []
    totals: dict[str, float] = {}
    for i, raw in enumerate(args.items, start=1):
        try:
            item = _Item(**raw)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
            )
            return f"Item {i} is invalid ({problems}). Fix it and call the tool again."

        if item.type == "per_diem":
            rate, currency = PER_DIEM_RATES[item.scope]
            payable = rate * item.quantity
            note = f"{item.quantity} day(s) x {rate} {currency} fixed per diem"
        else:
            if item.amount is None or item.currency is None:
                return f"Item {i} needs 'amount' and 'currency'. Fix it and call again."
            cap, cap_currency = REIMBURSEMENT_CAPS[(item.type, item.scope)]
            if item.currency != cap_currency:
                return (
                    f"Item {i}: {item.scope} {item.type} is limited in {cap_currency}, "
                    f"but the claim is in {item.currency}. Ask the user for the "
                    f"{cap_currency} amount; do not convert currencies."
                )
            currency = cap_currency
            per_unit = min(item.amount, cap)
            payable = per_unit * item.quantity
            note = f"{item.quantity} x min({item.amount:g}, cap {cap}) {currency}"
            if item.amount > cap:
                note += " (capped)"
        totals[currency] = totals.get(currency, 0) + payable
        lines.append(f"{i}. {item.scope} {item.type}: {payable:g} {currency} [{note}]")

    summary = ", ".join(f"{v:g} {cur}" for cur, v in totals.items())
    return "\n".join(lines) + f"\nTotal reimbursable: {summary}"


def list_leave_types(args: ListLeaveTypesInput) -> str:
    """List available leave types."""
    from src.retriever import retrieve_multi  # lazy: needs the index

    chunks = retrieve_multi(
        ["types of leave: vacation, sick leave, other leave", "أنواع الإجازات"],
        "hr",
        TOP_K,
    )
    return "Passages from the HR leave policies:\n\n" + _format_chunks(chunks)


TOOLS = {
    "policy_lookup": (PolicyLookupInput, policy_lookup),
    "reimbursement_calculator": (ReimbursementInput, reimbursement_calculator),
    "list_leave_types": (ListLeaveTypesInput, list_leave_types),
}


def run_tool(name: str, raw_args: dict) -> str:
    """Validate raw_args, run the tool, and return its text output.

    Never raises: unknown tools, bad input, and tool errors come back as
    feedback text so the agent can correct itself (one retry).
    """
    if name not in TOOLS:
        return f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."
    schema, func = TOOLS[name]
    try:
        args = schema(**(raw_args or {}))
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
        )
        return (
            f"Invalid input for {name}: {problems}. Fix the arguments and retry once."
        )
    try:
        return func(args)
    except FileNotFoundError as e:  # index missing
        return f"Tool {name} is unavailable: {e}"
    except Exception as e:  # noqa: BLE001 - keep the agent running
        return f"Tool {name} failed ({type(e).__name__}). Answer without it."
