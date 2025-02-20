import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from folding_api.chain import SubtensorService
from folding_api.protein import router
from folding_api.vars import (
    bt_config,
    limiter,
    logger,
    subtensor_service,
    validator_registry,
)

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def sync_metagraph_periodic(subtensor_service: SubtensorService):
    """Background task to sync metagraph every hour"""
    while True:
        try:
            logger.info("Syncing metagraph")
            # Run the synchronous function in a thread pool
            await asyncio.to_thread(subtensor_service.resync_metagraph)
            await asyncio.to_thread(validator_registry.update_validators)
            logger.info("Metagraph sync completed")
        except Exception as e:
            logger.error(f"Error syncing metagraph: {e}")

        await asyncio.sleep(600)


# Initialize API
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services
    logger.info("initializing_services")
    logger.info(f"Using config: {bt_config}")

    # Start background sync task
    app.state.sync_task = asyncio.create_task(
        sync_metagraph_periodic(subtensor_service)
    )
    logger.info("Started background sync task")

    yield

    # Cleanup
    logger.info("cleaning_up_services")
    if app.state.sync_task:
        logger.info("Canceling background sync task")
        app.state.sync_task.cancel()
        try:
            await app.state.sync_task
        except asyncio.CancelledError:
            pass


Instrumentator().instrument(app).expose(app)

# Include routes
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8030, reload=True)
