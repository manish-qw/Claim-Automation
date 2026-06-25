import os
import json
import logging
import requests
import google.generativeai as genai
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMService:
    """
    Hybrid LLM Router:
    - route_to_gemini(): Handles JSON-enforced reasoning tasks using Google Gemini 2.5 Flash
    - route_to_local(): Handles simple text summarization tasks using Ollama (Llama 3.2)
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.cache_ttl = 86400 * 30  # 30 days
        self.mode = os.environ.get("LLM_MODE", "disabled").lower()

        # Track if daily Gemini quota is exhausted per model — switching models resets it
        self._gemini_quota_exhausted: dict = {}  # {model_name: bool}

        # Gemini Config
        self.gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
        self.gemini_client = genai.GenerativeModel(self.gemini_model_name)

        # Local LLM Config (Ollama)
        self.local_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
        self.local_model = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.redis:
            return None
        try:
            val = self.redis.get(key)
            return val.decode('utf-8') if val else None
        except Exception as e:
            logger.warning(f"[LLMService] Redis GET failed: {e}")
            return None

    def _cache_set(self, key: str, value: str):
        if not self.redis:
            return
        try:
            self.redis.setex(self.cache_ttl, value)
        except Exception as e:
            logger.warning(f"[LLMService] Redis SET failed: {e}")

    def route_to_gemini(self, prompt: str, enforce_json: bool = True) -> Optional[Dict[str, Any]]:
        """
        Sends prompt to Gemini Flash.
        If enforce_json=True, it expects a JSON output and parses it.
        Gracefully returns None if API fails, letting callers use their algorithmic fallback.
        """
        if self.mode == "disabled":
            logger.info("[LLMService] LLM_MODE is disabled. Skipping Gemini call.")
            print("DEBUG: LLM_MODE is disabled. Skipping Gemini call.")
            return None

        # If daily quota already hit for this model, skip immediately
        if self._gemini_quota_exhausted.get(self.gemini_model_name, False):
            logger.info("[LLMService] Gemini daily quota exhausted for %s — skipping. Math fallback active.", self.gemini_model_name)
            print(f"DEBUG: _gemini_quota_exhausted is True for {self.gemini_model_name}")
            return None

        cache_key = f"llm:gemini:v1:{hash(prompt)}"
        cached = self._cache_get(cache_key)
        if cached:
            try:
                return json.loads(cached) if enforce_json else cached
            except json.JSONDecodeError:
                pass

        try:
            from google.api_core import retry as google_retry
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json" if enforce_json else "text/plain",
                temperature=0.0,  # Deterministic
            )
            # Allow retries so 429 Rate Limits (15 RPM free tier) succeed after brief wait
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={
                    "retry": google_retry.Retry(
                        initial=2.0, maximum=15.0, multiplier=2.0, deadline=60.0
                    )
                },
            )
            text_response = response.text.strip()

            if enforce_json:
                # Gemini sometimes still wraps response in ```json ... ``` even with application/json
                if text_response.startswith("```json"):
                    text_response = text_response[7:]
                if text_response.startswith("```"):
                    text_response = text_response[3:]
                if text_response.endswith("```"):
                    text_response = text_response[:-3]
                text_response = text_response.strip()

                result = json.loads(text_response)
                self._cache_set(cache_key, json.dumps(result))
                return result
            else:
                self._cache_set(cache_key, text_response)
                return text_response

        except Exception as e:
            err_str = str(e)
            print(f"DEBUG: Exception in route_to_gemini: {err_str}")
            # Detect daily quota exhaustion — stop ALL further Gemini calls this session
            if "GenerateRequestsPerDayPerProjectPerModel" in err_str or (
                "429" in err_str and "day" in err_str.lower()
            ):
                self._gemini_quota_exhausted[self.gemini_model_name] = True
                logger.warning(
                    "[LLMService] *** Gemini DAILY quota exhausted for %s. "
                    "All further Gemini calls for this model skipped. Math/rule engine handles all decisions. ***",
                    self.gemini_model_name
                )
                print(f"DEBUG: Quota exhausted flagged for {self.gemini_model_name}")
            else:
                logger.error(f"[LLMService] Gemini API call failed: {e}")
                print(f"DEBUG: Gemini API call failed: {e}")
            return None

    def route_to_local(self, prompt: str) -> Optional[str]:
        """
        Sends prompt to local Ollama instance for simple text generation.
        Gracefully returns None if local server is down.
        """
        if self.mode == "disabled":
            logger.info("[LLMService] LLM_MODE is disabled. Skipping Local LLM call.")
            return None

        cache_key = f"llm:local:v1:{hash(prompt)}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        payload = {
            "model": self.local_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
            },
        }

        try:
            response = requests.post(self.local_url, json=payload, timeout=120)  # 120s for long prompts
            response.raise_for_status()
            text_response = response.json().get("response", "").strip()
            self._cache_set(cache_key, text_response)
            return text_response
        except Exception as e:
            logger.error(f"[LLMService] Local LLM call failed (is Ollama running?): {e}")
            return None
