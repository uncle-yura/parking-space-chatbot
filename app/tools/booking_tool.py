import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from app.graphs.utils import RagGraphContextSchema


@tool
def start_booking_process() -> str:
    """Initiates the process of submitting a request to reserve a parking space."""
    return ""


@tool
def get_booking_status(runtime: ToolRuntime[RagGraphContextSchema]) -> str:
    """Check the status of the reservation created in the current conversation."""
    booking_id = runtime.state.get("booking_id")
    if booking_id is None:
        return "There is no reservation from this conversation to check."

    token = os.environ.get("ADMIN_SERVER_TOKEN")
    if not token:
        return "Reservation status is temporarily unavailable."

    url = os.environ.get("ADMIN_SERVER_URL", "http://localhost:8000")
    request = Request(
        f"{url.rstrip('/')}/booking-status/{booking_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urlopen(request, timeout=3) as response:
            booking = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return "The reservation could not be found."
        return "Reservation status is temporarily unavailable."
    except (URLError, TimeoutError, json.JSONDecodeError):
        return "Reservation status is temporarily unavailable."

    status = booking["status"]
    if status == "approved":
        return f"Reservation is approved. Assigned space: {booking['lot_identifier']}."
    if status == "rejected":
        return "Reservation was rejected because no space was available."
    if status == "pending":
        return "Reservation is still pending."
    return f"Reservation status: {status}."
