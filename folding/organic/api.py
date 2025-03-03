import uvicorn
from fastapi import FastAPI
from folding.base import validator
from folding.organic.organic import router as organic_router
from folding.utils.logging import logger

app = FastAPI()

app.include_router(organic_router)


async def start_organic_api(organic_validator, config):
    app.state.validator = organic_validator
    app.state.config = config

    logger.info(
        f"Starting organic API on  http://0.0.0.0:{config.neuron.organic_api.port}"
    )
    config = uvicorn.Config(
        "folding.organic.api:app",
        host="0.0.0.0",
        port=config.neuron.organic_api.port,
        loop="asyncio",
        reload=False,
    )
    server = uvicorn.Server(config)
    await server.serve()
