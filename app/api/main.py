# app/api/main.py
import uuid, time, logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from config import settings
from app.retrieval import Pipeline

log = logging.getLogger(__name__)
app = FastAPI(title="RAG-Bot API")
pipeline = Pipeline()


def _check_auth(req: Request):
    if (
        settings.API_KEY
        and req.headers.get("Authorization") != f"Bearer {settings.API_KEY}"
    ):
        raise HTTPException(401, "Unauthorized")


@app.post("/v1/chat/completions")
async def chat(req: Request):
    _check_auth(req)
    body = await req.json()
    query = body.get("messages", [{}])[-1].get("content")
    if not query:
        raise HTTPException(400, "`messages` array empty")

    stream = body.get("stream", False)
    chunks = pipeline.run(query)

    if stream:

        async def event_stream():
            for chunk in chunks:
                yield f"data: {chunk}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # non-stream
    answer = "".join(chunks)
    return JSONResponse(
        {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": settings.LLM_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
    )
