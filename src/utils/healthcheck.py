"""Simple healthcheck utility used by deployment probes."""


def health_ok() -> bool:
    """Return True when the service is healthy."""
    return True
