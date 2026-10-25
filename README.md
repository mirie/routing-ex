# Request Routing with LangChain and Gemini

A small Python learning project that uses Gemini to classify a request and route it to a simulated booking, information, or clarification handler.

The routing decision comes from a real Gemini API call. The handlers return demonstration messages; they do not make bookings or retrieve information.

Following along tutorial in Agentic Design Patterns / Antonio Gulli.

## Setup

You need uv, Python 3.12 or newer, and a Google API key with access to the configured Gemini model.

From the project directory, install the dependencies:

```bash
uv sync
```

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your key:

```dotenv
GOOGLE_API_KEY=your-google-api-key-here
```

`.env` is excluded from Git. Keep your actual key out of source files and commits.

## Run

```bash
uv run langchain_routing.py         # LangChain version
uv run python -m adk_routing.agent   # Google ADK version
uv run adk web                       # ADK browser UI (select adk_routing)
```

The program processes three examples in order:

| Request | Intended destination |
| --- | --- |
| Book me a flight to London. | `booker` |
| What is the capital of Italy? | `info` |
| I need to initiate a return for my item. | `unclear` |

For each request, it prints a heading, the selected handler's message, and the result. The example labels are for display; the model makes the routing decision. Running the examples sends requests to Google's API and is subject to your account's quotas and billing.

## How routing works

1. `load_dotenv()` loads the local configuration. The program exits if `GOOGLE_API_KEY` is missing.
2. `ChatGoogleGenerativeAI` creates the model client. The code currently selects `gemini-3.5-flash-lite`.
3. The prompt explains when to select each handler.
4. `with_structured_output(RoutingDecision, method="json_schema")` configures a structured response using the Pydantic schema.
5. `RunnablePassthrough.assign(decision=router)` preserves the original request and adds the parsed decision.
6. `process_request()` selects a function from the `handlers` dictionary and passes it the request text.

The data passed to handler selection has this shape:

```python
{
    "request": "Book me a flight to London.",
    "decision": RoutingDecision(destination="booker"),
}
```

`RoutingDecision` is a Pydantic model. Its `Literal["booker", "info", "unclear"]` annotation restricts the destination to those three values. Structured output controls the response format; it does not guarantee that the model chooses the right category.

## Code organization

The LangChain version is in `langchain_routing.py`:

- `RoutingDecision` defines the routing response schema.
- The three handler functions simulate specialist behavior.
- `Handler` is a type alias for a function that takes a string and returns a string.
- `process_request()` runs the coordinator and calls the selected handler.
- `main()` loads configuration, builds the routing chain, and runs the examples.

To experiment, edit the `requests` list, adjust the routing prompt, or change the handler functions. Change the `model` argument when you want to use another model available to your API key.

## Project files

| File | Purpose |
| --- | --- |
| `langchain_routing.py` | LangChain routing example and example requests |
| `adk_routing/agent.py` | Google ADK routing example (coordinator with sub-agents) |
| `.env.example` | Environment configuration template |

After editing dependencies in `pyproject.toml`, run `uv sync` to update the lockfile and environment. You can also use `uv add package-name` or `uv remove package-name` to manage dependencies directly.
