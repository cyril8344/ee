"""
auth.py
=======
JWT authentication helpers for the XAU/USD scalping bot API.

Uses only the Python standard library (hmac + hashlib) so there is no
dependency on the ``cryptography`` C extension.

Environment variables
---------------------
AUTH_SECRET      : JWT signing secret (random default per process if unset)
ALGORITHM        : (hardcoded) HS256
ADMIN_USERNAME   : login username (default: admin)
ADMIN_PASSWORD   : login password (default: changeme)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SECRET_KEY: str = os.environ.get("AUTH_SECRET", secrets.token_hex(32))
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS: int = 24

ADMIN_USERNAME: str = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD: str = os.environ.get("ADMIN_PASSWORD", "changeme")

bearer_scheme = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- #
# Pure-stdlib HS256 JWT helpers
# --------------------------------------------------------------------------- #
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    # Re-add padding
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return base64.urlsafe_b64decode(s)


def _sign(header_b64: str, payload_b64: str, secret: str) -> str:
    msg = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).digest()
    return _b64url_encode(sig)


# --------------------------------------------------------------------------- #
# Token helpers
# --------------------------------------------------------------------------- #
def create_access_token(data: dict) -> str:
    """Create a signed HS256 JWT that expires in ACCESS_TOKEN_EXPIRE_HOURS hours."""
    payload = data.copy()
    payload["exp"] = int(time.time()) + ACCESS_TOKEN_EXPIRE_HOURS * 3600
    payload["iat"] = int(time.time())

    header_b64 = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload_b64 = _b64url_encode(json.dumps(payload).encode())
    signature = _sign(header_b64, payload_b64, SECRET_KEY)
    return f"{header_b64}.{payload_b64}.{signature}"


def verify_token(token: str) -> dict:
    """
    Decode and validate a JWT.
    Raises HTTP 401 if the token is missing, malformed, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise credentials_exception

        header_b64, payload_b64, signature = parts

        # Verify signature
        expected_sig = _sign(header_b64, payload_b64, SECRET_KEY)
        if not hmac.compare_digest(expected_sig, signature):
            raise credentials_exception

        # Decode payload
        payload = json.loads(_b64url_decode(payload_b64))

        # Check expiry
        exp = payload.get("exp")
        if exp is None or int(time.time()) > exp:
            raise credentials_exception

        return payload
    except (ValueError, KeyError, UnicodeDecodeError):
        raise credentials_exception


# --------------------------------------------------------------------------- #
# FastAPI dependency
# --------------------------------------------------------------------------- #
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency — extract and verify the Bearer token from the
    Authorization header.  Returns the decoded payload dict.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_token(credentials.credentials)


# --------------------------------------------------------------------------- #
# Credential verification
# --------------------------------------------------------------------------- #
def verify_credentials(username: str, password: str) -> bool:
    """Return True if username + password match the configured admin account."""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD
