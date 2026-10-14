import time

import httpx
import requests

from config import MAX_RETRIES, MIN_INTERVAL

RETRY_STATUS = {429, 500, 502, 503, 504}
_last_call = 0.0


def with_retry(fn, attempts: int = MAX_RETRIES):
    """Exécute `fn()` en espaçant les appels ; réessaie avec un délai croissant.

    Réessaie sur 429/5xx (en respectant `Retry-After`) et sur les erreurs réseau.
    """
    global _last_call
    for attempt in range(1, attempts + 1):
        time.sleep(max(0.0, _last_call + MIN_INTERVAL - time.monotonic()))
        _last_call = time.monotonic()
        try:
            return fn()
        except (requests.HTTPError, httpx.HTTPStatusError) as exc:
            if exc.response.status_code not in RETRY_STATUS or attempt == attempts:
                raise
            retry_after = exc.response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else 2 ** attempt
        except (requests.ConnectionError, requests.Timeout, httpx.TransportError):
            if attempt == attempts:
                raise
            wait = 2 ** attempt
        print(f"[retry] nouvel essai dans {wait:.0f}s ({attempt}/{attempts - 1})")
        time.sleep(wait)
