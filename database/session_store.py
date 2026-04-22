# database/session_store.py

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from database.redis_client import RedisClient

logger = logging.getLogger(__name__)

# How long a user session lives in Redis without activity (7 days)
USER_SESSION_TTL_SECONDS = 7 * 24 * 3600


class UserSessionStore:
    """
    Persists per-user session data in Redis so users can resume
    exactly where they left off across logins and server restarts.

    Redis key: intelliquery:user_session:{email}

    Stored per user:
      - conversations  : list of {id, title, messages, charts, lastUpdated}
      - active_conv_id : which conversation was open when they left
      - last_active    : ISO timestamp of last activity
    """

    def __init__(self):
        self.redis = RedisClient.get_instance()
        logger.info(
            f"UserSessionStore initialized — "
            f"Redis: {'connected' if self.redis.is_connected else 'fallback mode'}"
        )

    # ------------------------------------------------------------------ #
    #  Key helper                                                          #
    # ------------------------------------------------------------------ #

    def _key(self, email: str) -> str:
        return f"intelliquery:user_session:{email}"

    # ------------------------------------------------------------------ #
    #  Core load / save                                                    #
    # ------------------------------------------------------------------ #

    def load_session(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Load full session for a user.
        Returns None if no session exists yet (first ever login).
        """
        data = self.redis.get(self._key(email))
        if data:
            self.redis.refresh_ttl(self._key(email), USER_SESSION_TTL_SECONDS)
            logger.info(f"✅ Session loaded for {email} — "
                        f"{len(data.get('conversations', []))} conversations")
        return data

    def save_session(self, email: str, session_data: Dict[str, Any]) -> bool:
        """
        Persist full session for a user.
        Call this after any mutation (new message, chart created, conv renamed).
        """
        session_data["last_active"] = datetime.now().isoformat()
        success = self.redis.set(
            self._key(email),
            session_data,
            ttl=USER_SESSION_TTL_SECONDS
        )
        if success:
            logger.info(f"💾 Session saved for {email}")
        else:
            logger.warning(f"⚠️ Failed to save session for {email} (Redis down?)")
        return success

    def delete_session(self, email: str) -> bool:
        """Delete a user's session (e.g. on explicit logout + clear)."""
        return self.redis.delete(self._key(email))

    def session_exists(self, email: str) -> bool:
        return self.redis.exists(self._key(email))

    # ------------------------------------------------------------------ #
    #  Conversation helpers (server-side mutations)                        #
    # ------------------------------------------------------------------ #

    def append_message(self, email: str, conv_id: int, message: Dict[str, Any]) -> bool:
        """
        Append a single message to a conversation.
        Used by /api/query to persist chat history server-side.
        """
        session = self.load_session(email)
        if not session:
            return False

        for conv in session.get("conversations", []):
            if conv["id"] == conv_id:
                conv.setdefault("messages", []).append(message)
                conv["lastUpdated"] = datetime.now().isoformat()
                return self.save_session(email, session)

        return False

    def append_chart(self, email: str, conv_id: int, chart: Dict[str, Any]) -> bool:
        """
        Append a chart to a conversation's visualization history.
        Used by /api/query when a chart is generated.
        """
        session = self.load_session(email)
        if not session:
            return False

        for conv in session.get("conversations", []):
            if conv["id"] == conv_id:
                conv.setdefault("charts", []).append(chart)
                conv["lastUpdated"] = datetime.now().isoformat()
                return self.save_session(email, session)

        return False

    def get_conversation(self, email: str, conv_id: int) -> Optional[Dict[str, Any]]:
        """Get a single conversation by ID."""
        session = self.load_session(email)
        if not session:
            return None
        for conv in session.get("conversations", []):
            if conv["id"] == conv_id:
                return conv
        return None

    # ------------------------------------------------------------------ #
    #  Stats / debug                                                       #
    # ------------------------------------------------------------------ #

    def get_session_stats(self, email: str) -> Dict[str, Any]:
        """Returns summary stats for a user's session — useful for debugging."""
        session = self.load_session(email)
        if not session:
            return {"exists": False}

        total_messages = sum(
            len(c.get("messages", [])) for c in session.get("conversations", [])
        )
        total_charts = sum(
            len(c.get("charts", [])) for c in session.get("conversations", [])
        )
        ttl = self.redis.get_ttl(self._key(email))

        return {
            "exists": True,
            "email": email,
            "conversations": len(session.get("conversations", [])),
            "total_messages": total_messages,
            "total_charts": total_charts,
            "last_active": session.get("last_active"),
            "ttl_seconds": ttl,
            "ttl_days": round(ttl / 86400, 1) if ttl > 0 else None,
        }


# Singleton instance
user_session_store = UserSessionStore()

__all__ = ["UserSessionStore", "user_session_store"]