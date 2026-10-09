# 💼 Agentic AI Finance Assistant

### An Evidence-Grounded Finance Agent with MCP, Policy RAG, Persistent Memory, and Security Validation

An end-to-end **Agentic AI Finance Assistant** built during my Agentic AI learning journey. This project explores how an LLM-powered agent can interact with structured financial data, perform deterministic calculations, reconcile financial transactions, retrieve policy evidence, and maintain conversational context using persistent memory.

Instead of relying on an LLM to answer everything from its internal knowledge, the system uses specialized tools, trusted data sources, retrieval validation, and persistent state to produce evidence-backed financial responses.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Agent Tools](#-agent-tools)
- [How the Agent Works](#-how-the-agent-works)
- [Memory and State Management](#-memory-and-state-management)
- [Policy RAG and Security](#-policy-rag-and-security)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Example Queries](#-example-queries)
- [Design Principles](#-design-principles)
- [Limitations and Future Enhancements](#-limitations-and-future-enhancements)
- [Learning Outcomes](#-learning-outcomes)

---

## 🎯 Project Overview

Financial analysis often involves working across multiple data sources, performing calculations, reconciling records, consulting internal policies, and retaining user-specific reporting preferences.

This project addresses these tasks through a **single tool-using AI agent** built with LangChain and LangGraph.

The agent can:

- Retrieve budget and actual financial records.
- Calculate budget-versus-actual variances.
- Reconcile bank statements against ledger records.
- Retrieve exchange rates and approval statuses through the Model Context Protocol (MCP).
- Search internal finance policies using Retrieval-Augmented Generation (RAG).
- Detect and quarantine suspicious instructions embedded in retrieved policy documents.
- Maintain conversation state through SQLite-backed checkpointing.
- Store and retrieve long-term user preferences.
- Summarize conversation history to manage context length.

### Project Objective

To design a modular, evidence-grounded, and security-conscious finance agent that combines deterministic financial operations with LLM-based reasoning, retrieval, and conversational memory.

**Core design principle:** The LLM determines which tools are appropriate and interprets their results, while financial calculations and data retrieval are handled by dedicated functions and trusted data sources.

---

## ✨ Key Features

| Feature | Implementation |
|---|---|
| Agent orchestration | LangChain `create_agent` |
| LLM integration | Groq API with `openai/gpt-oss-120b` |
| Financial data analysis | Pandas and structured CSV datasets |
| Deterministic calculations | Python `Decimal` arithmetic |
| Tool integration | LangChain tools |
| External tool protocol | Model Context Protocol (MCP) |
| Policy retrieval | Chroma vector store |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| Prompt-injection assessment | Structured LLM output with Pydantic |
| Short-term conversation state | LangGraph checkpointing with `AsyncSqliteSaver` |
| Long-term user preferences | SQLite |
| Context management | `SummarizationMiddleware` |
| Data persistence | SQLite databases and persistent Chroma storage |

---

## 🏗️ System Architecture

The architecture separates agent orchestration, trusted financial operations, MCP-based services, persistent memory, and policy retrieval into distinct components.

### Architecture Components

**1. Agent Orchestration**

The central agent coordinates user requests, dynamically incorporates preferences into its system prompt, manages conversation history, and decides when to invoke tools.

**2. Trusted Finance Tools**

Dedicated functions retrieve financial records, calculate variances, and reconcile bank and ledger transactions using deterministic logic.

**3. MCP Finance Operations**

A separate FastMCP server exposes exchange-rate lookup and approval-status lookup as tools that the agent can discover and invoke through `MultiServerMCPClient`.

**4. Policy RAG**

Finance policy documents are embedded and stored in Chroma. Relevant passages are retrieved for policy-related questions, assessed for suspicious instructions, and returned with source and page citations.

**5. Memory and Persistence**

SQLite stores long-term user preferences and agent checkpoints. Conversation history is recovered by thread ID, while preferences are associated with user IDs.

**6. Adaptive Evidence-Backed Responses**

The agent uses tool results and retrieved evidence to construct responses, following explicit rules against fabricated financial values, unsupported claims, and unauthorized actions.

---

## 🛠️ Technology Stack

### AI and Agent Frameworks
- LangChain
- LangGraph
- Groq API
- OpenAI GPT-OSS 120B model through Groq

### Retrieval and Security
- Chroma
- Hugging Face Sentence Transformers
- Pydantic structured output
- Retrieval-Augmented Generation (RAG)
- Prompt-injection assessment

### Data Engineering and Persistence
- Python
- Pandas
- SQLite
- `aiosqlite`
- Python `Decimal`
- JSON

### Tool Integration
- Model Context Protocol (MCP)
- FastMCP
- `MultiServerMCPClient`

---

## 🔧 Agent Tools

The agent combines locally defined finance tools with tools exposed through an MCP server.

### 1. Trusted Finance Tools

| Tool | Purpose |
|---|---|
| `query_financial_data` | Retrieves financial records filtered by reporting period and optionally by department and account. |
| `calculate_variance` | Calculates the absolute and percentage variance between actual and budget values. |
| `reconcile_accounts` | Compares bank and ledger transactions using reporting period, reference, and amount. |
| `search_finance_policies` | Retrieves relevant finance policy passages and returns screened evidence with citations. |

### 2. MCP Finance Operations

| Tool | Purpose |
|---|---|
| `get_exchange_rate` | Retrieves an exchange rate for a specified currency pair and date. |
| `get_approval_status` | Retrieves the status and owner of a finance approval request. |

The MCP server accesses exchange-rate and approval datasets through a dedicated SQLite database.

### Why use specialized tools?

Financial operations require predictable and verifiable behavior. Instead of asking the LLM to calculate values or invent database results, the agent delegates these operations to functions with defined inputs and outputs.

This separation improves traceability and reduces dependence on free-form LLM generation for numerical tasks.

---

## 🔄 How the Agent Works

### Step 1 — Receive the Request

The system receives a user ID, conversation thread ID, and natural-language question.

### Step 2 — Load User Preferences

The agent retrieves the user's saved preferences from SQLite, such as reporting precision or preferred response style.

### Step 3 — Construct the Dynamic Prompt

A dynamic prompt incorporates the base system instructions and applicable user preferences without allowing preferences to override security or evidence requirements.

### Step 4 — Manage Conversation Context

`SummarizationMiddleware` summarizes conversation history when the configured token threshold is reached, while retaining recent messages.

### Step 5 — Select and Invoke Tools

The LLM determines which tools are relevant to the request. Depending on the question, it can invoke financial data retrieval, variance calculation, reconciliation, MCP operations, or policy search.

### Step 6 — Validate Retrieved Policy Evidence

For policy-related requests, the RAG component retrieves relevant passages and assesses them for suspicious instructions before including them in the evidence returned to the agent.

### Step 7 — Generate the Response

The agent combines the question, tool results, and applicable policy evidence to produce a response that follows the system's evidence and security rules.

### Step 8 — Persist Conversation State

LangGraph checkpoints preserve the conversation state by thread ID, enabling the agent to continue a conversation across subsequent invocations.

### Step 9 — Update Long-Term Preferences

When a conversation is explicitly closed, the system extracts stable preferences from the saved user messages and updates the user's preference record in SQLite.

---

## 🧠 Memory and State Management

This project distinguishes between conversation state and long-term user preferences.

### Short-Term Memory: Conversation Checkpointing

**Technology:** LangGraph `AsyncSqliteSaver`

- Stores agent state associated with a conversation thread.
- Allows prior messages and state to be recovered across invocations.
- Uses a thread ID to identify a conversation.
- Uses an asynchronous SQLite connection for checkpoint persistence.

### Long-Term Memory: User Preferences

**Technology:** SQLite

- Associates preferences with a user ID rather than a conversation thread.
- Stores preferences as JSON.
- Supports inserting, updating, and removing explicitly changed preferences.
- Extracts stable preferences when a conversation is closed.

### Context Management: Summarization Middleware

**Technology:** LangChain `SummarizationMiddleware`

- Uses a configured token threshold to trigger summarization.
- Retains a specified number of recent messages.
- Helps control the size of conversation context sent to the model.

These mechanisms serve different purposes: checkpointing preserves conversational state, preference storage maintains user-specific settings, and summarization manages context length.

---

## 🔐 Policy RAG and Security

The project incorporates a policy retrieval pipeline with an additional validation step for retrieved content.

### Retrieval Pipeline

1. Load finance policy documents from text files.
2. Extract document content and source metadata.
3. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. Store and retrieve document representations through Chroma.
5. Retrieve the top four relevant passages for a query.
6. Assess each retrieved passage for suspicious instructions.
7. Quarantine passages classified as suspicious.
8. Return accepted evidence with source and page citations, alongside details of detected injection attempts.

### Prompt-Injection Assessment

The project uses a Pydantic model to structure the assessment result:

- `suspicious`: Whether the passage is classified as suspicious.
- `confidence`: A confidence value constrained to the range 0–1.
- `reason`: An explanation of the assessment.

The assessment is intended to identify passages that attempt to override instructions, request secrets, manipulate authorization, or trigger unauthorized tool use.

### Security Principles

- Retrieved documents are treated as untrusted input.
- User messages, preferences, and tool results are not treated as higher-priority instructions.
- The agent must not reveal credentials, hidden instructions, or unrelated data.
- Policy conclusions should be supported by retrieved evidence and citations.
- Suspicious retrieved passages must not be followed as instructions.
- The finance agent is designed for read-only operations and must not claim to approve, pay, post, or modify financial records.

*Security note: LLM-based prompt-injection assessment is a defense layer, not a guarantee that every malicious passage will be detected. Production deployments require additional controls and testing.*

---


### Runtime-Generated Files

The application creates or uses persistent state, including:

- `finance_operations_mcp.sqlite`
- `finance_agent_checkpoints.sqlite`
- `user_preferences.sqlite`
- `policy_chroma/`
- `finance_mcp_stderr.log`

These runtime artifacts should generally be excluded from version control. Keep the reproducible source files and safe sample datasets in Git, and create runtime databases locally.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 or a compatible version supported by the installed dependencies.
- A Groq API key with access to the configured model.
- The required finance CSV datasets.
- Finance policy text files.
- Internet access for the Groq API and initial model downloads.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd agentic-ai-finance-assistant
```

Replace the placeholder with your actual repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the dependencies used by the project:

```bash
pip install langchain langgraph langchain-groq langchain-chroma langchain-huggingface langchain-mcp-adapters mcp aiosqlite pandas pydantic sentence-transformers
```

For a reproducible setup, create and maintain a `requirements.txt` containing the versions validated in your environment.

### 4. Configure the API Key

Set the `GROQ_API_KEY` environment variable.

PowerShell:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key"
```

macOS or Linux:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

The application also includes an interactive fallback using `getpass` if the environment variable is not set.

**Never commit API keys, credentials, or populated environment files to GitHub.**

### 5. Add the Required Data and Policy Files

Place the required CSV files in:

```text
module4_finance_assets/data/
```

The application expects:

- `financials.csv`
- `ledger.csv`
- `bank.csv`
- `exchange_rates.csv`
- `approvals.csv`

Place policy documents in:

```text
module4_finance_assets/policies/
```

The current implementation reads policy files with a `.txt` extension and expects the required finance assets to be present before initialization.

Use synthetic or anonymized data when sharing this project publicly.

### 6. Run the Application

Execute the notebook or Python entry point that initializes the data, policy vector store, MCP server, checkpointing, and finance agent.

The exact command depends on how the repository is organized. Ensure the runtime working directory matches the location expected by `Path.cwd()` in the configuration.

---

## 💬 Example Queries

The following are representative requests supported by the implemented tool design. Exact responses depend on the supplied datasets and policy documents.

### Financial Data Retrieval

```text
What is the total budget for 2026-Q2?
```

The agent can retrieve the relevant financial records and use their values to construct an evidence-based response.

### Budget Variance Analysis

```text
Compare actual spending against the budget
for a specified department and reporting period.
```

The agent can retrieve the relevant records and calculate the absolute and percentage variance.

### Bank and Ledger Reconciliation

```text
Reconcile the bank statement with the ledger
for a specified reporting period.
```

The reconciliation tool identifies matched records and exceptions such as missing references or amount mismatches.

### Exchange Rate Lookup

```text
What is the exchange rate from EUR to USD
on a specified date?
```

The MCP exchange-rate tool retrieves the matching record when available.

### Approval Status

```text
Who owns approval request REQ-1001,
and what is its current status?
```

The MCP approval tool retrieves the status and owner from the approval dataset.

### Policy Retrieval

```text
What does the finance policy say about
a specified financial process?
```

The policy tool retrieves relevant passages and returns accepted evidence with source citations.

### Preference Persistence

```text
Report financial amounts to two decimal places
and remember this preference.
```

At conversation closure, the preference extraction process can save an explicitly stated, stable reporting preference for subsequent conversations.

---

## 📐 Design Principles

### Deterministic Operations over Unnecessary LLM Calls

Use ordinary code for calculations, filtering, and reconciliation. Reserve LLM calls for tasks that require language understanding, tool selection, structured extraction, or interpretation.

### Explicit Business Rules

Validate required columns, expected data grain, input parameters, and record-matching conditions before relying on financial results.

### Evidence over Assumptions

Financial values should come from data or deterministic calculations, while policy conclusions should be grounded in retrieved policy evidence.

### Separation of Responsibilities

Keep orchestration, business logic, retrieval, MCP services, memory, and security validation as distinct components.

### Persistence with Clear Ownership

Use thread IDs for conversational state and user IDs for long-term preferences.

### Security by Design

Treat retrieved documents and tool outputs as untrusted data. Restrict tools to their documented capabilities and avoid exposing secrets.

---

## 🚀 Limitations and Future Enhancements

The project is a learning implementation and is not yet a production-grade financial system.

Potential improvements include:

- **Automated testing:** Add unit tests for variance calculations, reconciliation edge cases, preference updates, and retrieval validation.
- **API layer:** Expose the agent through FastAPI and integrate a user interface.
- **Observability:** Add structured logging, tool-call tracing, latency measurements, and error monitoring.
- **Data validation:** Introduce explicit schemas, stronger input validation, and controls for duplicate or inconsistent records.
- **RAG evaluation:** Measure retrieval relevance, citation accuracy, and prompt-injection detection performance against a curated test set.
- **Security hardening:** Add access controls, authorization checks, tool allowlists, and adversarial testing.
- **Scalability:** Evaluate a server-grade database and persistent vector infrastructure if usage grows.
- **Human oversight:** Introduce approval workflows for any future financial write operations rather than allowing autonomous execution.
- **Deployment:** Containerize the application and implement CI/CD after establishing a reliable test suite.

---

## 🎓 Learning Outcomes

Building this project provided hands-on experience with several important Agentic AI engineering concepts:

- Designing a single agent that selects and invokes specialized tools.
- Integrating external capabilities using the Model Context Protocol.
- Combining structured data operations with retrieval-augmented generation.
- Managing short-term conversation state and long-term user preferences.
- Implementing dynamic prompts and context summarization middleware.
- Using structured LLM output for validation workflows.
- Designing evidence-backed responses with source attribution.
- Applying security principles to retrieved content and tool execution.
- Understanding the importance of deterministic business logic in AI applications.

The main takeaway is that building an effective AI agent involves more than connecting an LLM to tools. It requires carefully designed workflows, reliable data access, explicit business rules, state management, validation, and security controls.

---

## 👨‍💻 Author

**Pranesh Krishnan**

This project was developed as part of my Agentic AI learning journey to explore practical patterns for building stateful, tool-using AI systems.

---

⭐ If you find this project useful, consider starring the repository and sharing feedback or suggestions for improvement.
