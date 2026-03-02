"""Error response models for OpenAPI documentation."""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str
