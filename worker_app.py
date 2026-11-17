import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from azure.cosmos.exceptions import CosmosHttpResponseError
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from farmer_service import iso, utcnow
from main import farmer_service
from whatsapp_worker import WhatsAppWorker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    worker = WhatsAppWorker(farmer_service)
    app.state.last_tick = None
    app.state.healthy = True

    async def consume():
        while True:
            try:
                await asyncio.to_thread(worker.tick)
                app.state.last_tick = iso(utcnow())
                app.state.healthy = True
            except CosmosHttpResponseError:
                logger.exception("Cosmos worker tick failed; retrying in one minute.")
                app.state.healthy = False
            await asyncio.sleep(60)

    task = asyncio.create_task(consume())
    app.state.worker_task = task
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    healthy = app.state.healthy and not app.state.worker_task.done()
    return JSONResponse(
        {"status": "running" if healthy else "worker_error", "lastTick": app.state.last_tick},
        status_code=200 if healthy else 503,
    )
