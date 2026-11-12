from __future__ import annotations

import copy
from typing import Annotated

from fastapi import Depends, HTTPException, Query

from synneon.services.memory.store import ContextStore

# Global store instance — set during server lifespan startup
_store: ContextStore | None = None


def set_context_store(store: ContextStore) -> None:
    global _store
    _store = store


def get_context_store(
    user_id: str = Query("default", description="User ID namespace for this request."),
) -> ContextStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Context store not initialized")
    if user_id == _store._user_id:
        return _store
    # Shallow copy sharing the Redis connection — only _user_id is scoped per request
    scoped = copy.copy(_store)
    scoped._user_id = user_id
    return scoped


ContextStoreDep = Annotated[ContextStore, Depends(get_context_store)]
