from langchain_core.tools import tool


@tool
def start_booking_process() -> str:
    """Initiates the process of submitting a request to reserve a parking space."""
    return ""