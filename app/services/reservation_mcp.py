import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("parking-reservations")
_write_lock = threading.Lock()
logger = logging.getLogger(__name__)


@mcp.tool()
def save_approved_reservation(
    name: str, car_number: str, reservation_start: str, reservation_end: str
) -> str:
    """Persist reservation details after administrator approval only."""
    if any(
        "\n" in value or "\r" in value or "|" in value
        for value in (name, car_number, reservation_start, reservation_end)
    ):
        raise ValueError("Reservation fields cannot contain pipes or newlines.")

    path = Path(os.environ.get("RESERVATIONS_FILE", "/data/approved_reservations.txt"))
    reservation_period = f"{reservation_start} - {reservation_end}"
    line = f"{name} | {car_number} | {reservation_period} | {datetime.now(UTC).isoformat()}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving approved reservation.")
    with _write_lock:
        if path.exists():
            for existing_line in path.read_text(encoding="utf-8").splitlines():
                fields = [field.strip() for field in existing_line.split("|")]
                if fields[:3] == [name, car_number, reservation_period]:
                    return "Approved reservation already saved."

        with path.open("a", encoding="utf-8") as records:
            records.write(line)
            records.flush()
            os.fsync(records.fileno())

    return "Approved reservation saved."
