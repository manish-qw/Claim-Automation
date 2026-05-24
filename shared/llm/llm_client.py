"""
shared/llm/llm_client.py
─────────────────────────────────────────────────────────────────────────────
PURPOSE:
    Unified LLM client used by every agent in the system.
    No agent imports the Anthropic or OpenAI SDK directly — all LLM
    calls go through this module.

WHAT GOES HERE:

    LLMModel (enum):
        OPUS    — claude-opus-4   (used for: Fraud Debate, Decision, Policy RAG)
        SONNET  — claude-sonnet-4 (used for: Intake, Synthesis, Communications)

    LLMResponse (dataclass):
        text          — the model's response string
        model_used    — actual model string that handled the request
        tokens_used   — {input_tokens, output_tokens}
        prompt_version — version tag passed in
        latency_ms    — end-to-end latency in milliseconds

    FUNCTIONS:

    complete(prompt: str, system: str, model: LLMModel,
             prompt_version: str) → LLMResponse
        Standard text completion (no tools). Internally handles all
        resilience logic listed below.

    complete_with_tools(prompt: str, system: str, tools: list,
                        model: LLMModel) → LLMResponse
        Tool-use completion for ReAct-style agents. Returns tool call
        results alongside text.

    INTERNAL BEHAVIOUR (implemented inside this module):

    1. PII TOKENISATION (DPDP Act 2023 compliance — mandatory):
            Before every API call, scans the prompt for:
                - Aadhaar numbers (12-digit sequences)
                - PAN numbers (AAAAA9999A format)
                - Bank account numbers
                - Full names (from claim context)
            Replaces each with a reversible placeholder token:
                e.g. "7412 5896 3210" → "<AADHAAR_TOKEN_1>"
            After response returns, re-substitutes tokens back
            with the original values in the response text.
            Token map is kept in memory for the duration of the call only.

    2. RETRY WITH EXPONENTIAL BACKOFF:
            On rate limit (429) or server error (5xx): retry up to 3 times.
            Backoff schedule: 2s → 4s → 8s between attempts.

    3. PROVIDER FAILOVER:
            If Anthropic API returns 503 after all 3 retries:
            automatically falls back to OpenAI GPT-4o with the same prompt.
            Logs the fallover event as a WARNING in audit.

    4. AUTOMATIC AUDIT LOGGING:
            Every LLM call automatically logs an AuditEvent via
            audit_service.log() — no agent needs to do this separately.
            Logged fields: model_used, prompt_version, tokens_used,
            latency_ms, input_hash, output_hash.

DEPENDENCIES:
    anthropic, openai, shared.audit.audit_service, shared.config.settings
"""
