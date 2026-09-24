import os
import csv
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, TXTSearchTool

try:
    import gspread
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="CrewAI Support Dashboard",
    page_icon="🤖",
    layout="centered"
)

# Target Google Sheet URL
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1HBD6iHd4X7S2BPbHs_FSZLw6d-C6YBXoLkJuTPfF1GY/edit?gid=0#gid=0"

# Directories setup
docs_path = os.path.join("knowledge", "docs.txt")
os.makedirs("knowledge", exist_ok=True)
os.makedirs("output", exist_ok=True)

# App Title & UI Header
st.title("🤖 Multi-Agent Customer Support")
st.caption("Powered by CrewAI (RAG -> Web Search -> Google Sheets Logging)")

# Sidebar Navigation & Information
with st.sidebar:
    st.header("📌 System Info")
    st.markdown("""
    **Agents Active:**
    1. **Agent 1:** Local RAG Specialist (`knowledge/docs.txt`)
    2. **Agent 2:** Serper Web Search Fallback
    3. **Agent 3:** Sheet Resolution Logger
    """)
    st.divider()
    st.link_button("📊 View Live Google Sheet", SPREADSHEET_URL)

# Helper Function: Google Sheet Upload via service_account.json
def log_to_google_sheet(timestamp, inquiry, summary, source):
    if not GSPREAD_AVAILABLE:
        return False, "gspread library not installed."

    current_dir = os.path.dirname(os.path.abspath(__file__))
    service_account_path = os.path.join(current_dir, "service_account.json")

    if not os.path.exists(service_account_path):
        return False, f"Missing 'service_account.json' file at {service_account_path}"

    try:
        gc = gspread.service_account(filename=service_account_path)
        sh = gc.open_by_url(SPREADSHEET_URL)
        worksheet = sh.sheet1

        existing_rows = worksheet.get_all_values()
        if len(existing_rows) == 0:
            worksheet.append_row(["Timestamp", "Customer Inquiry", "Answer / Resolution", "Source / Agent Used"])

        worksheet.append_row([timestamp, inquiry, summary, source])
        return True, "Successfully logged to Google Sheet!"
    except Exception as e:
        return False, str(e)

# Form Component for Customer Query Input
with st.form("support_form"):
    user_inquiry = st.text_area("Enter Customer Inquiry:", placeholder="e.g., How do I handle a damaged item return?")
    submit_button = st.form_submit_button("Submit Query", type="primary")

# Execute Pipeline on Form Submission
if submit_button and user_inquiry.strip():
    with st.spinner("🤖 CrewAI Agents are processing your inquiry..."):
        
        # Initialize LLM and Tools
        llm = LLM(model="gpt-4o-mini", temperature=0.3)
        web_search_tool = SerperDevTool()
        rag_tool = TXTSearchTool(txt=docs_path)

        # AGENT 1: Internal Knowledge Base Specialist (RAG)
        rag_agent = Agent(
            role="Internal Knowledge Base Specialist",
            goal="Answer customer inquiries accurately using internal documentation stored in the knowledge directory.",
            backstory=(
                "You search local company files in the knowledge directory to answer queries. "
                "If internal documentation lacks the answer, output ONLY: 'NEEDS_WEB_SEARCH'."
            ),
            tools=[rag_tool],
            llm=llm
        )

        # AGENT 2: Web Research Specialist (Serper Search Fallback)
        web_agent = Agent(
            role="Web Research Support Specialist",
            goal="Perform live internet searches using Serper search tool to answer customer questions.",
            backstory=(
                "You step in when internal documentation is insufficient. "
                "When Agent 1 flags 'NEEDS_WEB_SEARCH', execute your search tool to gather live information."
            ),
            tools=[web_search_tool],
            llm=llm
        )

        # AGENT 3: Support Interaction Data Logger
        sheet_logger_agent = Agent(
            role="Support Interaction Data Logger",
            goal="Extract and format key details into a single-sentence resolution summary for spreadsheet entry.",
            backstory="You analyze customer inquiries and final responses to create clean summaries.",
            llm=llm
        )

        # TASK 1: Local RAG Search
        rag_task = Task(
            description=(
                f"Search local knowledge base files for an answer to this inquiry:\n'{user_inquiry}'\n"
                "If answer is found, compose response. If missing, output ONLY: 'NEEDS_WEB_SEARCH'."
            ),
            expected_output="Customer support response grounded in local docs OR 'NEEDS_WEB_SEARCH'.",
            agent=rag_agent
        )

        # TASK 2: Serper Web Search Fallback
        web_task = Task(
            description=(
                f"Review output from Task 1. If Task 1 outputted 'NEEDS_WEB_SEARCH', "
                f"perform a web search for: '{user_inquiry}' and provide a complete response."
            ),
            expected_output="A complete customer support response based on live web search results.",
            agent=web_agent,
            context=[rag_task]
        )

        # TASK 3: Resolution Summary Logging
        logging_task = Task(
            description=(
                f"Review customer inquiry ('{user_inquiry}') and final response from Task 2. "
                "Create a clean single-sentence summary of the resolution for spreadsheet logging."
            ),
            expected_output="A clean single-sentence summary of the final resolution.",
            agent=sheet_logger_agent,
            context=[rag_task, web_task]
        )

        # Assemble and Run 3-Agent Sequential Crew
        crew = Crew(
            agents=[rag_agent, web_agent, sheet_logger_agent],
            tasks=[rag_task, web_task, logging_task],
            process=Process.sequential
        )

        result = crew.kickoff()

        # Data Preparation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_summary = str(result.raw).replace("\n", " ").strip()
        source = "Web Search" if "NEEDS_WEB_SEARCH" in str(rag_task.output) else "Internal Docs"

        # Local CSV Backup Logging
        csv_path = os.path.join("output", "support_sheet_logs.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Customer Inquiry", "Answer / Resolution", "Source / Agent Used"])
            writer.writerow([timestamp, user_inquiry, clean_summary, source])

        # Google Sheets Upload
        sheet_success, sheet_msg = log_to_google_sheet(timestamp, user_inquiry, clean_summary, source)

    # Render Results in Streamlit Frontend
    st.success("Query Processed Successfully!")
    
    st.markdown("### 💬 Final Response")
    st.info(result.raw)

    st.markdown("### 📊 Logging Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Source Used", value=source)
    with col2:
        if sheet_success:
            st.success("Synced to Google Sheet!")
        else:
            st.warning(f"Saved Locally (Sheet Note: {sheet_msg})")