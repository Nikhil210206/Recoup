"""Razorpay webhook signature verification.

Razorpay signs the webhook with HMAC-SHA256 over the *raw request body* using
the webhook secret, and sends the hex digest in `X-Razorpay-Signature`.

Two details matter and both are easy to get wrong:

  - The HMAC must be computed over the exact bytes received. Parsing the JSON
    and re-serialising it changes whitespace and key order, and the signature
    stops matching for reasons that look like a Razorpay bug.

  - The comparison must be constant-time. `==` on a digest leaks timing
    information that can be used to forge a signature byte by byte.
"""

from __future__ import annotations

import hashlib
import hmac


def compute_signature(raw_body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def verify(raw_body: bytes, signature: str | None, secret: str) -> bool:
    """Return True only if the signature is present, the secret is configured,
    and the digest matches. Missing config fails closed."""
    if not signature or not secret:
        return False
    expected = compute_signature(raw_body, secret)
    return hmac.compare_digest(expected, signature)
