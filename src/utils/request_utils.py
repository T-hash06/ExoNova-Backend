import uuid
from datetime import datetime, timezone


def generate_request_id() -> str:
    """
    Generate a unique request identifier.
    
    Uses UUID4 for generating cryptographically strong random identifiers
    with extremely low probability of collision.
    
    Returns:
        str: Unique request identifier in format 'req_<uuid4>'
    """
    return f"req_{uuid.uuid4().hex}"


def get_timestamp() -> str:
    """
    Get current timestamp in ISO 8601 format.
    
    Returns UTC timestamp with timezone information for consistent
    time representation across different systems and timezones.
    
    Returns:
        str: Current timestamp in ISO 8601 format (e.g., '2025-10-06T12:34:56.789Z')
    """
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
