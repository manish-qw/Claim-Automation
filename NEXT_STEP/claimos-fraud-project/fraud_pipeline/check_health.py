"""
check_health.py -- Quick health check for all pipeline services
Run anytime: python check_health.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results = {}

# -- 1. Gemini API -------------------------------------------------------------
print("Checking Gemini API...", end=" ", flush=True)
try:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model_name)
    r = client.generate_content("Reply with the single word: OK")
    results["Gemini API"] = f"[OK]  WORKING  (model: {model_name})"
except Exception as e:
    err = str(e)
    if "429" in err:
        results["Gemini API"] = "[!!] QUOTA EXCEEDED -- resets at 5:30 AM IST (midnight UTC)"
    elif "404" in err:
        results["Gemini API"] = f"[X]  MODEL NOT FOUND -- check GEMINI_MODEL in .env (got: {model_name})"
    elif "API_KEY" in err or "invalid" in err.lower():
        results["Gemini API"] = "[X]  INVALID API KEY -- check GEMINI_API_KEY in .env"
    else:
        results["Gemini API"] = f"[X]  ERROR -- {err[:120]}"

# -- 2. Ollama / Local LLM -----------------------------------------------------
print("Checking Ollama (Local LLM)...", end=" ", flush=True)
try:
    import requests
    local_model = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
    r = requests.post(
        os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/api/generate"),
        json={"model": local_model, "prompt": "Say OK", "stream": False},
        timeout=60
    )
    r.raise_for_status()
    results["Ollama / Llama"] = f"[OK]  WORKING  (model: {local_model})"
except Exception as e:
    err = str(e)
    if "404" in err:
        local_model = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
        results["Ollama / Llama"] = f"[X]  MODEL NOT PULLED -- run: ollama pull {local_model}"
    elif "10061" in err or "Connection refused" in err or "actively refused" in err:
        results["Ollama / Llama"] = "[X]  OLLAMA NOT RUNNING -- run: ollama serve"
    elif "timed out" in err.lower():
        results["Ollama / Llama"] = "[!!] TIMEOUT -- model is loading, try again in 30s"
    else:
        results["Ollama / Llama"] = f"[X]  ERROR -- {err[:120]}"

# -- 3. OLS API (EMBL-EBI Ontology) --------------------------------------------
print("Checking OLS API...", end=" ", flush=True)
try:
    import requests
    r = requests.get(
        "https://www.ebi.ac.uk/ols4/api/search?q=cardiac+arrest&ontology=mondo&rows=1",
        timeout=10
    )
    r.raise_for_status()
    count = r.json().get("response", {}).get("numFound", 0)
    results["OLS API (EMBL-EBI)"] = f"[OK]  WORKING  ({count} results for test query)"
except Exception as e:
    results["OLS API (EMBL-EBI)"] = f"[X]  UNREACHABLE -- {str(e)[:100]}"

# -- 4. Neo4j ------------------------------------------------------------------
print("Checking Neo4j...", end=" ", flush=True)
try:
    from neo4j import GraphDatabase
    uri  = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd  = os.environ.get("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    driver.close()
    results["Neo4j"] = f"[OK]  WORKING  ({uri})"
except Exception:
    results["Neo4j"] = "[--] NOT RUNNING  (OK -- pipeline uses NetworkX fallback)"

# -- 5. PostgreSQL -------------------------------------------------------------
print("Checking PostgreSQL...", end=" ", flush=True)
try:
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        database=os.environ.get("POSTGRES_DB", "claimos_fraud"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        connect_timeout=3
    )
    conn.close()
    results["PostgreSQL"] = "[OK]  WORKING"
except Exception:
    results["PostgreSQL"] = "[--] NOT RUNNING  (OK -- pipeline uses neutral frequency fallback)"

# -- 6. Redis ------------------------------------------------------------------
print("Checking Redis...", end=" ", flush=True)
try:
    import redis
    rc = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        socket_connect_timeout=2
    )
    rc.ping()
    results["Redis"] = "[OK]  WORKING  (LLM response caching active)"
except Exception:
    results["Redis"] = "[--] NOT RUNNING  (OK -- caching disabled, pipeline still works)"

# -- Print Results -------------------------------------------------------------
print("\n")
print("=" * 62)
print("  CLAIMOS Pipeline Health Check")
print("=" * 62)
for service, status in results.items():
    print(f"  {service:<22}  {status}")
print("=" * 62)

critical_down = any(
    "[X]" in v
    for k, v in results.items()
    if k in ("Gemini API", "Ollama / Llama")
)
quota_hit = "[!!]" in results.get("Gemini API", "")

if quota_hit:
    print("\n  [!!] Gemini quota hit. Wait until 5:30 AM IST then re-run.")
    print("       Pipeline still works using math rules + Llama 3.2.\n")
elif critical_down:
    print("\n  [X]  Fix the items above before running the pipeline.\n")
else:
    print("\n  All systems ready. Run:  python run_demo.py\n")
