"""Fernet symmetric encryption for API keys."""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet

from exo_web.config import settings


def _derive_key(secret: str) -> bytes:
    """Derive a 32-byte Fernet key from the app secret_key using SHA-256."""
    digest = hashlib.sha256(secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def _get_fernet() -> Fernet:
    """Return a Fernet instance using the app secret_key."""
    return Fernet(_derive_key(settings.secret_key))


def encrypt_api_key(plaintext: str) -> str:
    """Encrypt an API key and return the ciphertext as a string."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    """Decrypt an encrypted API key and return the plaintext."""
    return _get_fernet().decrypt(ciphertext.encode()).decode()
