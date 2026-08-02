"""
Smart Learning Hub - Chatbot Microservice (Python / FastAPI)

Replaces the old approach where the Spring Boot backend called the
Gemini/OpenAI APIs directly (see ai/GeminiAiProvider.java,
ai/OpenAiAiProvider.java). This service now owns the LLM API keys and the
provider-specific HTTP calls; the Java backend only builds the prompt
(with course/lesson context, pulled from MySQL) and forwards it here.

Run locally:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health        -> liveness check
    POST /chat           -> { "prompt": "..." } -> { "reply": "...", "provider": "gemini" }
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse
from app.providers import get_active_provider

app = FastAPI(
    title="Smart Learning Hub - Chatbot Service",
    description="Python microservice that talks to the LLM provider (Gemini/OpenAI) on behalf of the Spring Boot backend.",
    version="1.0.0",
)

# Only the Spring Boot backend should call this service directly (server-to-server),
# but CORS is left open here since it sits behind the backend, not the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "UP", "service": "chatbot-service"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    provider = get_active_provider()

    try:
        reply = provider.generate_response(request.prompt)
    except Exception as exc:  # keep the service alive, surface a clean error to Java
        raise HTTPException(status_code=502, detail=f"Upstream AI provider error: {exc}")

    return ChatResponse(reply=reply, provider=provider.name)
