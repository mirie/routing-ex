import logging
import os
from collections.abc import Callable
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


Handler = Callable[[str], str]


class RoutingDecision(BaseModel):
    destination: Literal["booker", "info", "unclear"]


# --- Define Simulated Subagent Handlers (equivalent to ADK sub_agents) ---
def booking_handler(request: str) -> str:
    """Simulates the Booking Agent handling a request."""
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action."


def info_handler(request: str) -> str:
    """Simulates the Info Agent handling a request"""
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval."


def unclear_handler(request: str) -> str:
    """Handles requests that couldn't be delegated."""
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


def process_request(request: str, coordinator: Runnable, handlers: dict[str, Handler]) -> str:
    """Route the request to the appropriate handler and return its response."""
    data = coordinator.invoke({"request": request})
    handler = handlers[data["decision"].destination]
    return handler(data["request"])


def main() -> None:
    # --- Configuration ---
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY in your environment or .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
    )

    # -- Define Coordinator prompt
    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
        """Analyze the user's request and determine which specialist handler should process it.
        - Requests to make a flight or hotel booking, choose 'booker'.
        - For questions seeking information, choose 'info'.
        - If the request is unclear or doesn't fit either category, choose 'unclear'.
        """),
        ("user", "{request}")
    ])

    router = router_prompt | llm.with_structured_output(
        RoutingDecision,
        method="json_schema",
    )

    coordinator = RunnablePassthrough.assign(decision=router)

    handlers = {
        "booker": booking_handler,
        "info": info_handler,
        "unclear": unclear_handler
    }

    requests = [
        {
            "label": "booking",
            "request": "Book me a flight to London."
        },
        {
            "label": "info",
            "request": "What is the capital of Italy?"
        },
        {
            "label": "unclear",
            "request": "I need to initiate a return for my item."
        },
    ]

    for r in requests:
        print(f"\n\n--- Running with a {r['label']} request ---")

        try:
            result = process_request(r['request'], coordinator, handlers)
        except Exception:
            logging.exception("Failed to process request: %s", r['request'])
            continue

        print(f"Result: {result}")

if __name__ == "__main__":
    main()
