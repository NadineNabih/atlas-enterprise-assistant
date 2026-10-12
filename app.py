"""Chat UI entry point (Chainlit). Owner: Person 3.

Run: chainlit run app.py
Must show: domain, retrieved sources, tool calls, cited answer, RTL Arabic,
and a New chat control.
"""

import json
import time

import chainlit as cl

from src.graph import run_question
from src.logging_setup import chunk_metadata, log_event, new_run_id
from src.memory import clear_session
from src.settings import AR_RATIO_ARABIC_MIN, AR_RATIO_ENGLISH_MAX

DOMAIN_LABELS = {"hr": "HR", "it": "IT", "finance": "Finance"}


def detect_language(text: str) -> str:
    """Return "en", "ar" or "bilingual" using the shared settings thresholds."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    ratio = sum("\u0600" <= ch <= "\u06ff" for ch in letters) / len(letters)
    if ratio < AR_RATIO_ENGLISH_MAX:
        return "en"
    return "ar" if ratio > AR_RATIO_ARABIC_MIN else "bilingual"


@cl.set_starters
async def starters(user=None):
    """Example questions shown on an empty chat (one per domain, EN and AR)."""
    return [
        cl.Starter(
            label="Vacation days", message="How many vacation days do new hires get?"
        ),
        cl.Starter(label="VPN setup", message="What is the VPN server and port?"),
        cl.Starter(
            label="Hotel limit", message="What is the hotel cap for domestic travel?"
        ),
        cl.Starter(label="إجازة مرضية", message="كم عدد أيام الإجازة المرضية السنوية؟"),
    ]


@cl.on_chat_start
async def start():
    """A new chat (including the built-in New Chat button) gets a clean session."""
    await cl.Message(
        content="Atlas Enterprise Assistant is ready. / مساعد أطلس جاهز."
    ).send()


@cl.on_chat_end
async def end():
    """Forget the conversation memory when the chat is closed or restarted."""
    try:
        clear_session(cl.user_session.get("id"))
    except NotImplementedError:
        pass  # memory.py is not ready yet


@cl.on_message
async def main(message: cl.Message):
    question = message.content
    session_id = cl.user_session.get("id")
    language = detect_language(question)
    fallback_run_id = new_run_id()
    started = time.perf_counter()

    try:
        state = await cl.make_async(run_question)(question, session_id)
    except NotImplementedError:
        log_event(fallback_run_id, "ui_error", error="graph_not_ready")
        await cl.Message(
            content="The agent graph is not connected yet (part 2)."
        ).send()
        return
    except Exception as e:  # noqa: BLE001 - never show a stack trace to the user
        log_event(
            fallback_run_id, "ui_error", error=type(e).__name__, question=question
        )
        await cl.Message(
            content="Sorry, something went wrong. Please try again. / حدث خطأ، حاول مرة أخرى."
        ).send()
        return

    run_id = state.get("run_id", fallback_run_id)
    domain = state.get("domain", "")
    tool_calls = state.get("tool_calls", [])
    chunks = state.get("retrieved_chunks", [])
    sources = state.get("sources", [])

    for call in tool_calls:  # tool calls
        async with cl.Step(name=call.name, type="tool") as step:
            step.input = json.dumps(call.input, ensure_ascii=False)
            step.output = call.output

    # One side panel per source file, showing the passage that was used.
    elements, seen = [], set()
    for chunk in chunks:
        if chunk.source_filename in sources and chunk.source_filename not in seen:
            seen.add(chunk.source_filename)
            elements.append(
                cl.Text(name=chunk.source_filename, content=chunk.text, display="side")
            )

    header = f"**Domain:** {DOMAIN_LABELS.get(domain, domain or 'n/a')}\n\n"
    sources_md = ""
    if sources:
        sources_md = "\n\n**Sources:** " + ", ".join(sources)
    await cl.Message(
        content=header + state.get("answer", "") + sources_md, elements=elements
    ).send()

    log_event(
        run_id,
        "ui_response",
        question=question,
        language=language,
        domain=domain,
        tools=[c.name for c in tool_calls],
        retrieved=chunk_metadata(chunks),
        sources=sources,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
