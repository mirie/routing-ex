import asyncio
import logging
import uuid

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

load_dotenv()


# --- Define Tool Functions ---
# These functions simulate the actions of the specialist agents
def booking_handler(request: str) -> str:
    """
    Simulates the Booking Agent handling a request.
    Args:
        request: The user's booking request.
    Returns:
        A confirmation message that the booking was handled.
    """
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action."


def info_handler(request: str) -> str:
    """
    Simulates the Info Agent handling a request.
    Args:
        request: The user's information inquiry.
    Returns:
        A confirmation message that the information retrieval was handled.
    """
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval."


def unclear_handler(request: str) -> str:
    """
    Handles requests that couldn't be delegated.
    Args:
        request: The user's input.
    Returns:
        A message noting that the request could not be handled.
    """
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


# --- Create tools from functions ---
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)
unclear_tool = FunctionTool(unclear_handler)

# Define specialized sub-agents equipped with their respective tools.
booking_agent = Agent(
    name="Booker",
    model="gemini-3.5-flash-lite",
    description="A specialized agent that handles all flight and hotel booking requests by calling the booking tool.",
    tools=[booking_tool],
)

info_agent = Agent(
    name="Info",
    model="gemini-3.5-flash-lite",
    description="A specialized agent that handles information queries and answers user questions by calling the info tool.",
    tools=[info_tool],
)

# Define the parent agent with delegation instructions
coordinator = Agent(
    name="Coordinator",
    model="gemini-3.5-flash-lite",
    instruction=(
        "You are the main coordinator. Your only task is to analyze incoming user requests "
        "and delegate them to the appropriate specialist agent. "
        "Do not try to answer the user directly.\n"
        "- For any requests related to booking flights or hotels, delegate to the 'Booker' agent.\n"
        "- For general information questions, delegate to the 'Info' agent.\n"
        "- If the request fits neither, call the unclear_handler tool."
    ),
    description="A coordinator that routes user requests to the correct specialist agent.",
    tools=[unclear_tool],
    # delegation is handled via ADK's Auto-flow mechanism
    sub_agents=[booking_agent, info_agent],
)

# `adk web` / `adk run` look for a module-level variable named root_agent.
root_agent = coordinator


# --- Execution logic ---
async def run_coordinator(runner: InMemoryRunner, request: str) -> str:
    """Runs the coordinator agent with a given request and delegates."""
    print(f"\n--- Running Coordinator with request: '{request}' ---")

    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=request)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_result = "".join(part.text for part in event.content.parts if part.text)
                break

        print(f"Coordinator Final Response: {final_result}")
        return final_result

    except Exception as e:
        print(f"An error occurred while processing the request: {e}")
        return f"An error occurred while processing request: {e}"


async def main():
    """Main function to run the ADK example."""
    print("--- Google ADK routing example -- with Auto-flow style ---")
    runner = InMemoryRunner(coordinator)

    requests = [
        {"label": "booking", "request": "Book me a flight to London."},
        {"label": "info", "request": "What is the capital of Italy?"},
        {"label": "unclear", "request": "I need to initiate a return for my item."},
    ]

    for r in requests:
        print(f"\n\n--- Running with a {r['label']} request ---")
        try:
            result = await run_coordinator(runner, r["request"])
            print(f"Final input - Result: {result}")
        except Exception:
            logging.exception("Failed to process request: %s", r["request"])
            continue


if __name__ == "__main__":
    asyncio.run(main())
