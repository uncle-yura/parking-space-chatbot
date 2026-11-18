import logging
import os
import secrets
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.graphs.administrator_graph import AdministratorState, build_administrator_graph
from app.graphs.utils import AdministratorGraphContextSchema
from app.services.llm_chat import OllamaLLMService
from app.services.sql_service import SqlService

logger = logging.getLogger(__name__)
app = FastAPI(title="Parking Reservation Processor")
bearer_scheme = HTTPBearer(auto_error=False)
bearer_dependency = Depends(bearer_scheme)


async def _run_processor() -> None:
    try:
        graph = await build_administrator_graph()
        state = await graph.compile().ainvoke(
            AdministratorState(messages=[], pending=[], processed=[]),
            context=AdministratorGraphContextSchema(
                sql_service=SqlService(), llm=OllamaLLMService().llm
            ),
        )
        logger.info(state)
    except Exception:
        logger.exception("Background reservation processing failed")


@app.get("/health")
def health():
    return {"status": "ok"}


def _authorize(credentials: HTTPAuthorizationCredentials | None) -> None:
    expected_token = os.environ.get("ADMIN_SERVER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Administrator authentication is not configured.",
        )
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, expected_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.post("/booking-processing", status_code=status.HTTP_202_ACCEPTED)
def start_booking_processing(
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials | None = bearer_dependency,
):
    _authorize(credentials)
    background_tasks.add_task(_run_processor)
    return {"status": "accepted", "message": "Reservation processing started."}


@app.get("/booking-status/{booking_id}")
def booking_status(
    booking_id: UUID,
    credentials: HTTPAuthorizationCredentials | None = bearer_dependency,
):
    _authorize(credentials)
    booking = SqlService().get_booking_status(booking_id)
    if booking is None:
        raise HTTPException(status_code=404, detail="Reservation not found.")
    return booking
