# CLAIMOS & Policy Agent Integration - Full Project Setup Guide

This guide provides step-by-step instructions to set up, download dependencies, and run the entire CLAIMOS Fraud Intelligence Pipeline along with the Policy Orchestrator.

The project consists of two main components:
1. **CLAIMOS Fraud Project (`claimos-fraud-project`)**: A 7-agent AI pipeline running on FastAPI that detects insurance fraud using a hybrid approach of local and cloud LLMs, backed by Postgres, Redis, and Neo4j.
2. **Policy Agent (`policy_agent`)**: A LangGraph-based policy interpretation orchestrator that takes the output of the fraud pipeline and reasons over policy documents to make a final decision.

---

## 🛠️ Prerequisites

Before you start, ensure you have the following installed on your system (Windows OS):

1. **Python 3.10+**: Make sure Python is added to your system PATH.
2. **Docker Desktop**: Required to run the infrastructure (PostgreSQL, Redis, Neo4j). Ensure Docker Desktop is installed and running in the background.
3. **Ollama**: Required to run the local Llama 3.2 model for privacy-safe summarization.
   - Download from [Ollama's Official Website](https://ollama.com/).

---

## 🚀 Complete Step-by-Step Setup

### Step 1: Install and Configure Local AI (Ollama)
The Fraud Pipeline uses a local Llama model for Agent 6 to generate executive summaries.

1. Install Ollama and open a PowerShell terminal.
2. Pull the required Llama 3.2 model:
   ```powershell
   ollama pull llama3.2
   ```
3. Ensure Ollama is running in the background.

### Step 2: Configure Environment Variables (.env)
You need to set up API keys (like Google Gemini) for the cloud-based AI reasoning.

1. Navigate to `claimos-fraud-project\fraud_pipeline`.
2. Create or edit the `.env` file and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   LLM_MODE=HYBRID
   ```
3. Navigate to `policy_agent`.
4. Ensure the `.env` file there also contains the required API keys (e.g., `GEMINI_API_KEY`, depending on your policy agent configuration).

### Step 3: Setup Python Virtual Environment and Dependencies
It is highly recommended to isolate your project dependencies using a virtual environment.

4. Install all required dependencies (this covers both projects as `policy_agent` uses `claimos-fraud-project` libraries):
   ```powershell
   cd claimos-fraud-project
   pip install -r fraud_pipeline\requirements.txt
   ```

### Step 4: Start the Infrastructure & Backend (Fraud Pipeline)
Now, we need to spin up the databases (Postgres, Neo4j, Redis) and the FastAPI backend.

1. Make sure Docker Desktop is running.
2. Ensure your Python virtual environment is activated in PowerShell.
3. While inside the `claimos-fraud-project` directory, execute the startup script:
   ```powershell
   .\start.ps1
   ```
   *This script will automatically:*
   - Start `postgres`, `redis`, and `neo4j` Docker containers.
   - Wait for Postgres to be ready.
   - Launch the FastAPI server on `http://localhost:8001`.

*Leave this terminal open and running. Press `CTRL+C` to stop the server when you are done.*

### Step 5: Run the Full Pipeline (Integration)
With the backend server running, you can now execute the full pipeline that connects the Fraud Engine to the Policy Orchestrator.

1. Open a **new** PowerShell terminal.

3. Navigate to the `policy_agent` directory:
   ```powershell
   cd d:\workspace\policy_agent
   ```
4. Run the full pipeline! You have two options:

   **Option A: Run with built-in Mock Data**
   ```powershell
   python run_full_pipeline.py
   ```
   
   **Option B: Run with a Live Claim JSON payload**
   ```powershell
   python run_full_pipeline.py --live test_case_1_full.json
   ```

---

## 🎯 Verification & API Usage

- **API Documentation**: Once `start.ps1` is running, you can view the FastAPI Swagger documentation at [http://localhost:8001/docs](http://localhost:8001/docs).
- **Health Check**: Verify the backend is running at [http://localhost:8001/fraud/health](http://localhost:8001/fraud/health).

If you want to bypass the Policy Agent and test just the Fraud Pipeline, you can send a POST request directly to the API:
```powershell
# Using PowerShell
Invoke-RestMethod -Uri "http://localhost:8001/fraud/analyze" -Method Post -Body '{"claim_case_id": "CLM-123", ...}' -ContentType "application/json"
```

## 🛑 Troubleshooting
- **Docker Errors**: If `start.ps1` fails at the Docker step, ensure Docker Desktop is open and running properly. You can try running `docker compose up -d` manually in the `claimos-fraud-project\infrastructure` folder.
- **LLM Timeouts**: If you hit rate limits with Gemini, the pipeline is designed to degrade gracefully to offline algorithms.
- **Import Errors**: Ensure you have activated your `venv` in *every* terminal window you open before running python commands.
