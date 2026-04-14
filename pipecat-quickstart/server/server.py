import asyncio
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from bot import run_bot

load_dotenv(override=True)

webrtc_connections: dict[str, SmallWebRTCConnection] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Spell Bee server starting on http://0.0.0.0:7860")
    yield
    logger.info(f"Closing {len(webrtc_connections)} active connection(s)...")
    for conn in list(webrtc_connections.values()):
        await conn.close()
    webrtc_connections.clear()
    logger.info("Server shut down cleanly")


app = FastAPI(
    title="Spell Bee Voice Bot",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/connect")
async def connect(offer: dict):
    connection_id = offer.get("id", "default")
    logger.info(f"New /connect request — connection_id: {connection_id}")

    if connection_id not in webrtc_connections:
        conn = SmallWebRTCConnection()
        webrtc_connections[connection_id] = conn

        await conn.initialize(
            sdp=offer["sdp"],
            type=offer["type"],
        )

        asyncio.create_task(run_bot(conn))

        await conn.connect()
    else:
        conn = webrtc_connections[connection_id]

        await conn.renegotiate(
            sdp=offer["sdp"],
            type=offer["type"],
            restart_pc=offer.get("restart_pc", False),
        )

    return conn.get_answer()


@app.delete("/connect/{connection_id}")
async def disconnect(connection_id: str):
    if connection_id in webrtc_connections:
        await webrtc_connections[connection_id].close()
        del webrtc_connections[connection_id]
        logger.info(f"Connection {connection_id} removed")
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)