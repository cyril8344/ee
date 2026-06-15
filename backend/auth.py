"""
auth.py
=======
JWT authentication helpers for the XAU/USD scalping bot API.

Environment variables
---------------------
AUTH_SECRET      : JWT signing secret (random default per process if unset)
ALGORITHM        : (hardcoded) HS256
ADMIN_USERNAME   : login username (default: admin)
ADMIN_PASSWORD   : login password (default: changeme)
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SECRET_KEY: str = os.environ.get("AUTH_SECRET", secrets.token_hex(32))
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS: int = 24

ADMIN_USERNAME: str = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD: str = os.environ.get("ADMIN_PASSWORD", "changeme")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- #
# Token helpers
# --------------------------------------------------------------------------- #
def create_access_token(data: dict) -> str:
    """Create a signed JWT that expires in ACCESS_TOKEN_EXPIRE_HOURS hours."""
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


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
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
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
