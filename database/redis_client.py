# database/redis_client.py

import redis
import json
import os
import logging
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class RedisClient:
    """
    Singleton Redis client for IntelliQuery.
    Handles connection, serialization, TTL, and graceful fallback to in-memory
    if Redis is unavailable so the app never crashes.
    """

    _instance: Optional["RedisClient"] = None

    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.password = os.getenv("REDIS_PASSWORD") or None
        self.db = int(os.getenv("REDIS_DB", 0))
        self.ttl_seconds = int(os.getenv("REDIS_TTL_HOURS", 24)) * 3600
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._connect()

    @classmethod
    def get_instance(cls) -> "RedisClient":
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = RedisClient()
        return cls._instance

    def _connect(self):
        """Attempt to connect to Redis. Sets _connected flag — never raises."""
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True,       # Always return strings, not bytes
                socket_connect_timeout=3,    # Fail fast if Redis is down
                socket_timeout=3,
                retry_on_timeout=True,
            )
            self._client.ping()
            self._connected = True
            logger.info(f"✅ Redis connected at {self.host}:{self.port} (db={self.db})")
        except Exception as e:
            self._connected = False
            logger.warning(
                f"⚠️  Redis unavailable at {self.host}:{self.port} — {e}. "
                f"Running in fallback in-memory mode."
            )

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    #  Core key-value operations                                           #
    # ------------------------------------------------------------------ #

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store any JSON-serializable value with TTL.
        Returns True on success, False on failure.
        default=str handles datetime / Decimal values from PostgreSQL.
        """
        if not self._connected:
            return False
        try:
            serialized = json.dumps(value, default=str)
            self._client.setex(
                name=key,
                time=ttl if ttl is not None else self.ttl_seconds,
                value=serialized,
            )
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve and deserialize a value.
        Returns None if key doesn't exist or on error.
        """
        if not self._connected:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True on success."""
        if not self._connected:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error for key '{key}': {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if not self._connected:
            return False
        try:
            return bool(self._client.exists(key))
        except Exception:
            return False

    def refresh_ttl(self, key: str, ttl: Optional[int] = None) -> bool:
        """
        Reset the expiry timer on a key.
        Call this on every user activity to keep active sessions alive.
        """
        if not self._connected:
            return False
        try:
            self._client.expire(key, ttl if ttl is not None else self.ttl_seconds)
            return True
        except Exception:
            return False

    def get_ttl(self, key: str) -> int:
        """
        Returns remaining TTL in seconds.
        -1 = no expiry, -2 = key doesn't exist.
        """
        if not self._connected:
            return -2
        try:
            return self._client.ttl(key)
        except Exception:
            return -2

    def get_keys_by_pattern(self, pattern: str) -> list:
        """
        List all keys matching a pattern.
        e.g. get_keys_by_pattern('intelliquery:dashboard:*')
        """
        if not self._connected:
            return []
        try:
            return self._client.keys(pattern)
        except Exception:
            return []

    def flush_all_intelliquery_keys(self) -> int:
        """
        Delete all IntelliQuery keys (useful for dev/testing resets).
        Returns number of keys deleted.
        """
        if not self._connected:
            return 0
        try:
            keys = self._client.keys("intelliquery:*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis flush error: {e}")
            return 0

    # ------------------------------------------------------------------ #
    #  Namespaced key helpers — keeps the keyspace organized               #
    # ------------------------------------------------------------------ #

    def dashboard_key(self, session_id: str) -> str:
        """intelliquery:dashboard:{session_id}"""
        return f"intelliquery:dashboard:{session_id}"

    def agent_state_key(self, session_id: str) -> str:
        """intelliquery:agent_state:{session_id}"""
        return f"intelliquery:agent_state:{session_id}"

    def last_query_key(self, session_id: str) -> str:
        """intelliquery:last_query:{session_id}"""
        return f"intelliquery:last_query:{session_id}"

    def conversation_key(self, user_email: str) -> str:
        """intelliquery:conversation:{user_email}"""
        return f"intelliquery:conversation:{user_email}"

    # ------------------------------------------------------------------ #
    #  Debug helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """
        Returns a summary of current Redis state.
        Useful for the /health endpoint.
        """
        if not self._connected:
            return {"status": "unavailable", "mode": "in-memory fallback"}
        try:
            info = self._client.info("stats")
            keyspace = self._client.info("keyspace")
            intelliquery_keys = len(self._client.keys("intelliquery:*"))
            return {
                "status": "connected",
                "host": f"{self.host}:{self.port}",
                "db": self.db,
                "intelliquery_keys": intelliquery_keys,
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace": keyspace,
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}