from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Optional


class AgentEvent(BaseModel):
    event_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str           # "query", "filter", "rank", "explain", "result"
    user_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True
