from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from ratelimit import limits, RateLimitException
from tools.logging_handler import EnterpriseLogger
import re

class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 1000, rate_limit: str = "100/hour"):
        super().__init__(app)
        self.max_requests = max_requests
        self.rate_limit = rate_limit
        self.logger = EnterpriseLogger().get_logger(__name__)

    async def dispatch(self, request: Request, call_next):
        try:
            self.validate_headers(request)
            self.validate_content_type(request)
            self.apply_rate_limits(request)
            
            self.logger.info(
                "api_request",
                method=request.method,
                path=request.url.path,
                client=request.client.host
            )
            
            response = await call_next(request)
            return response
        
        except RateLimitException as e:
            return JSONResponse(status_code=429, content={"detail": "Too many requests"})
        except HTTPException as he:
            self.logger.error("api_error", status=he.status_code, detail=he.detail)
            raise
        except Exception as ex:
            self.logger.critical("unhandled_exception", error=str(ex))
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    def validate_headers(self, request: Request):
    # Always require User-Agent
        required_headers = ["User-Agent"]
        
        # Add Content-Type only for methods with request bodies
        if request.method in ("POST", "PUT", "PATCH"):
            required_headers.append("Content-Type")
        
        for header in required_headers:
            if header not in request.headers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required header: {header}"
                )

    def validate_content_type(self, request: Request):
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("Content-Type", "")
            
            # Allow multipart/form-data for file uploads
            if not (
                re.match(r"^application/json", content_type) or
                re.match(r"^multipart/form-data", content_type)
            ):
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported media type"
                )

    @limits(calls=100, period=3600)
    def apply_rate_limits(self, request: Request):
        pass