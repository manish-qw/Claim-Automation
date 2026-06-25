"""
Redis service — caching, temporary workflow state, queues.
"""

from __future__ import annotations
import json
from typing import Any, Optional

from utils.config import settings
from utils.logger import get_logger

logger = get_logger("RedisService")

CACHE_TTL_SECONDS = 3600  # 1 hour


class RedisService:
    def __init__(self):
        try:
            import redis
            self._client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
            self._client.ping()
            self._available = True
            logger.info("RedisService connected")
        except Exception as e:
            self._available = False
            self._client = None
            logger.warning(f"Redis unavailable (caching disabled): {e}")

    def set_claim_result(self, claim_case_id: str, data: dict, ttl: int = CACHE_TTL_SECONDS) -> bool:
        if not self._available:
            return False
        try:
            key = f"fraud_result:{claim_case_id}"
            self._client.setex(key, ttl, json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
            return False

    def get_claim_result(self, claim_case_id: str) -> Optional[dict]:
        if not self._available:
            return None
        try:
            key = f"fraud_result:{claim_case_id}"
            val = self._client.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        return None

    def enqueue_claim(self, queue_name: str, claim_case_id: str) -> bool:
        if not self._available:
            return False
        try:
            self._client.rpush(queue_name, claim_case_id)
            return True
        except Exception as e:
            logger.warning(f"Redis enqueue failed: {e}")
            return False
