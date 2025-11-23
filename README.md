# FlightPlannerAI
Flight Planner AI is an intelligent assistant built with LangChain and a GPT-based model. It can search flights, book tickets, and handle complaints by saving them to files. Users can interact via a web interface or command-line, making travel planning simple, interactive, and automated.
# FlightPlannerAI (local mock mode supported)

Quick start (Windows PowerShell):

1. Create a virtual environment

```powershell
python -m venv .venv
```

2. Install minimal dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. (Optional) Copy `.env.example` to `.env` and add `LANGCHAIN_API_KEY` if you want full LangChain/OpenAI features.

```powershell
copy .env.example .env
notepad .env
```

If you want LangSmith run-tracking, also add the `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` values into `.env`.

Optional AI dependencies

If you want the full LangChain/OpenAI experience (requires more packages), install the optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
```

This keeps the default install fast while allowing you to opt into heavy AI packages when you're ready.

4. Run the app (local mock mode will be used if no API key or LangChain is not installed):

```powershell
.\.venv\Scripts\python.exe flight_planner.py
```

Local mock usage (single text input):
- `search|ORIGIN|DESTINATION|YYYY-MM-DD`
- `book|FLIGHT_NO|PASSENGER_NAME`
- `complaint|Your complaint text here`

To enable full LangChain/OpenAI features, install the optional packages and set `LANGCHAIN_API_KEY` in `.env`.
# Byte-compiled / optimized / DLL files
# flight_planner.py
import os
from dotenv import load_dotenv
from datetime import datetime
import json
from typing import Any, Dict, List, Optional

import gradio as gr

# LangChain/OpenAI imports are optional. If unavailable or no API key
# is provided, the app runs in a lightweight local/mock mode that
# uses the dummy tools below without contacting external APIs.
HAS_LANGCHAIN = True
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.agents import Tool, initialize_agent
    from langchain.agents import AgentType
except Exception:
    HAS_LANGCHAIN = False

# -----------------------------
# Load LangSmith / API credentials
# -----------------------------
load_dotenv()  # Make sure you have a .env file with LANGCHAIN_API_KEY or Gemini creds
API_KEY = os.getenv("LANGCHAIN_API_KEY", "")  # Empty means run mock/local mode

# If LANGCHAIN_API_KEY is missing or imports failed, we'll run in local mock mode.
USE_LOCAL_MOCK = (not API_KEY) or (not HAS_LANGCHAIN)

# If the user provided LangSmith/OpenAI keys, expose them to the environment
# so LangChain / LangSmith integrations can pick them up.
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_KEY)

LANGSMITH_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_KEY)

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
if LANGSMITH_PROJECT:
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

# -----------------------------
# Dummy flight search tool
# -----------------------------
def search_flights(origin: str, destination: str, date: str) -> str:
    """Dummy flight search"""
    # Normally you would call a real API here
    available_flights = [
        {"flight_no": "AB123", "time": "10:00", "price": 120},
        {"flight_no": "CD456", "time": "15:30", "price": 150},
    ]
    # If the date is in the past, no flights
    if datetime.strptime(date, "%Y-%m-%d") < datetime.now():
        return "No flights available for the selected date."
    return json.dumps(available_flights, indent=2)

# -----------------------------
# Dummy flight booking tool
# -----------------------------
def book_flight(flight_no: str, passenger_name: str) -> str:
    """Dummy flight booking"""
    confirmation = {
        "flight_no": flight_no,
        "passenger": passenger_name,
        "status": "Booked",
        "confirmation_code": "XYZ789"
    }
    return f"Flight booked successfully:\n{json.dumps(confirmation, indent=2)}"

# -----------------------------
# Complaint tool
# -----------------------------
def raise_complaint(complaint_text: str) -> str:
    """Save complaint to a txt file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"complaint_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(complaint_text)
    return f"Complaint saved to {filename}"

# -----------------------------
# LangChain Tools
# -----------------------------
if not USE_LOCAL_MOCK:
    # -----------------------------
    # LangChain-backed agent (requires API key and langchain installed)
    # -----------------------------
    tools = [
        Tool(
            name="Search Flights",
            func=search_flights,
            description="Search available flights given origin, destination, and date (format YYYY-MM-DD)."
        ),
        Tool(
            name="Book Flight",
            func=book_flight,
            description="Book a flight using flight number and passenger name."
        ),
        Tool(
            name="Raise Complaint",
            func=raise_complaint,
            description="Save a complaint text to a file."
        )
    ]

    # Chat model
    chat_model = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful Flight Planner Assistant. 
You have access to the following tools:
- Search Flights: Search available flights.
- Book Flight: Book a flight for a passenger.
- Raise Complaint: Save a complaint text to a file.

Use the tools to fulfill user requests. Answer politely.
User Input: {input}
"""
    )

    agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    def run_agent(user_input):
        try:
            result = agent.run(user_input)
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    iface = gr.Interface(
        fn=run_agent,
        inputs="text",
        outputs="text",
        title="Flight Planner AI",
        description="Enter flight requests, bookings, or complaints."
    )

else:
    # -----------------------------
    # Local/mock mode (no external AI). Keep interface but use the
    # simple dummy functions directly. This is useful for development
    # or when you don't want to install LangChain/OpenAI.
    # Supported input formats (single text input):
    #   search|ORIGIN|DESTINATION|YYYY-MM-DD
    #   book|FLIGHT_NO|PASSENGER_NAME
    #   complaint|Your complaint text here
    # -----------------------------
    def run_agent(user_input: str) -> str:
        try:
            parts = [p.strip() for p in user_input.split("|")]
            cmd = parts[0].lower() if parts else ""
            if cmd == "search" and len(parts) == 4:
                origin, dest, date = parts[1], parts[2], parts[3]
                return search_flights(origin, dest, date)
            elif cmd == "book" and len(parts) == 3:
                flight_no, passenger = parts[1], parts[2]
                return book_flight(flight_no, passenger)
            elif cmd == "complaint" and len(parts) >= 2:
                complaint_text = "|".join(parts[1:])
                return raise_complaint(complaint_text)
            else:
                return (
                    "Local mock mode — use one of these commands:\n"
                    "search|ORIGIN|DESTINATION|YYYY-MM-DD\n"
                    "book|FLIGHT_NO|PASSENGER_NAME\n"
                    "complaint|Your complaint text here\n"
                )
        except Exception as e:
            return f"Error (local mode): {e}"

    iface = gr.Interface(
        fn=run_agent,
        inputs="text",
        outputs="text",
        title="Flight Planner AI (Local Mock Mode)",
        description=(
            "Local/mock mode. No external API calls.\n"
            "Use commands: search|ORIGIN|DESTINATION|YYYY-MM-DD,\n"
            "book|FLIGHT_NO|PASSENGER_NAME, complaint|TEXT"
        )
    )


if __name__ == "__main__":
    iface.launch()
    # If you use LangSmith-specific SDKs add them here
python-dotenv
gradio
python-dotenv
gradio

# Optional heavy AI dependencies (move to requirements-ai.txt)
# langchain
# openai
Search Flights

Input:

search|New York|Los Angeles|2025-12-25


Output:

[
  {
    "flight_no": "AB123",
    "time": "10:00",
    "price": 120
  },
  {
    "flight_no": "CD456",
    "time": "15:30",
    "price": 150
  }
]

