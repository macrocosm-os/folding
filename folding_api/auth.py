import json
import os
import uuid
from typing import Dict, Optional, List
from fastapi import HTTPException, Request, APIRouter, Depends
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from folding_api.schemas import APIKey, APIKeyCreate, APIKeyResponse


class APIKeyManager:
    def __init__(self, api_key_file: str = "api_keys.json"):
        # check if the file exists
        self.api_key_file = api_key_file
        self.api_keys: Dict[str, APIKey] = {}
        self.api_key_header = APIKeyHeader(name="X-API-Key")
        if not os.path.exists(api_key_file):
            with open(api_key_file, "w") as f:
                json.dump({}, f)

            self.create_api_key("admin", "1000/hour")
        self.load_api_keys()

    def load_api_keys(self):
        """Load API keys from file"""
        try:
            with open(self.api_key_file, "r") as f:
                data = json.load(f)
                self.api_keys = {key: APIKey(**value) for key, value in data.items()}
        except FileNotFoundError:
            self.api_keys = {}

    def save_api_keys(self):
        """Save API keys to file"""
        with open(self.api_key_file, "w") as f:
            json.dump(
                {k: v.model_dump() for k, v in self.api_keys.items()}, f, indent=4
            )

    def create_api_key(self, owner: str, rate_limit: str = "100/hour") -> str:
        """Create a new API key"""
        key = str(uuid.uuid4())
        self.api_keys[key] = APIKey(key=key, owner=owner, rate_limit=rate_limit)
        self.save_api_keys()
        return key

    def get_api_key(self, key: str) -> Optional[APIKey]:
        """Get API key details"""
        return self.api_keys.get(key)

    def validate_api_key(self, key: str) -> bool:
        """Validate an API key"""
        api_key = self.api_keys.get(key)
        if not api_key or not api_key.is_active:
            return False
        return True

    def deactivate_api_key(self, key: str):
        """Deactivate an API key"""
        if key in self.api_keys:
            self.api_keys[key].is_active = False
            self.save_api_keys()

    def get_rate_limit(self, key: str) -> Optional[str]:
        """Get rate limit for an API key"""
        api_key = self.api_keys.get(key)
        if api_key:
            return api_key.rate_limit
        return None


async def get_api_key(
    request: Request, api_key: str = Depends(APIKeyHeader(name="X-API-Key"))
) -> APIKey:
    """Dependency for FastAPI to validate API keys"""
    api_key_obj = request.app.state.api_key_manager.get_api_key(api_key)
    if not api_key_obj or not api_key_obj.is_active:
        raise HTTPException(status_code=403, detail="Invalid or inactive API key")
    return api_key_obj


async def get_admin_api_key(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    """Dependency to ensure the API key has admin privileges"""
    if api_key.owner != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return api_key


def create_api_key_limiter(api_key: APIKey) -> Limiter:
    """Create a rate limiter for an API key"""
    return Limiter(key_func=get_remote_address, default_limits=[api_key.rate_limit])


# Create router for API key management
api_key_router = APIRouter(prefix="/api-keys", tags=["API Keys"])


@api_key_router.post("", response_model=APIKeyResponse)
async def create_api_key(
    request: Request, api_key_data: APIKeyCreate, _: APIKey = Depends(get_admin_api_key)
) -> APIKeyResponse:
    """Create a new API key (admin only)"""
    key = request.app.state.api_key_manager.create_api_key(
        owner=api_key_data.owner, rate_limit=api_key_data.rate_limit
    )
    api_key = request.app.state.api_key_manager.get_api_key(key)
    return APIKeyResponse(**api_key.model_dump())


@api_key_router.get("", response_model=List[APIKeyResponse])
async def list_api_keys(
    request: Request, _: APIKey = Depends(get_admin_api_key)
) -> List[APIKeyResponse]:
    """List all API keys (admin only)"""
    return [
        APIKeyResponse(**key.model_dump())
        for key in request.app.state.api_key_manager.api_keys.values()
    ]


@api_key_router.get("/me", response_model=APIKeyResponse)
async def get_current_api_key(api_key: APIKey = Depends(get_api_key)) -> APIKeyResponse:
    """Get details of the current API key"""
    return APIKeyResponse(**api_key.model_dump())


@api_key_router.delete("/{key}")
async def deactivate_api_key(
    key: str, request: Request, _: APIKey = Depends(get_admin_api_key)
):
    """Deactivate an API key (admin only)"""
    if not request.app.state.api_key_manager.get_api_key(key):
        raise HTTPException(status_code=404, detail="API key not found")
    request.app.state.api_key_manager.deactivate_api_key(key)
    return {"message": "API key deactivated successfully"}
