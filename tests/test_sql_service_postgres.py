import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier
from urllib.parse import unquote, urlparse
from uuid import UUID

import psycopg
import pytest

from app.services.sql_service import SqlService

pytestmark = pytest.mark.integration


def _test_database_url():
    url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "Set POSTGRES_TEST_DATABASE_URL to run PostgreSQL integration tests"
        )
    database_name = urlparse(url).path.strip("/")
    if not (database_name.startswith("test_") or database_name.endswith("_test")):
        pytest.fail(
            "POSTGRES_TEST_DATABASE_URL must point to a database named with a 'test_' prefix or '_test' suffix"
        )
    return url


@pytest.fixture
def sql_service(monkeypatch):
    url = _test_database_url()
    parsed = urlparse(url)
    monkeypatch.setenv("POSTGRES_HOST", parsed.hostname or "localhost")
    monkeypatch.setenv("POSTGRES_PORT", str(parsed.port or 5432))
    monkeypatch.setenv("POSTGRES_USER", unquote(parsed.username or "postgres"))
    monkeypatch.setenv("POSTGRES_PASSWORD", unquote(parsed.password or ""))
    monkeypatch.setenv("POSTGRES_DB", parsed.path.strip("/"))
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_lots (
                id SERIAL PRIMARY KEY,
                identifier TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                customer_name TEXT NOT NULL,
                car_number TEXT NOT NULL,
                start_at TIMESTAMPTZ NOT NULL,
                end_at TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
                lot_id INTEGER REFERENCES parking_lots(id) ON DELETE SET NULL,
                customer_id INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                review_claimed_at TIMESTAMPTZ,
                mcp_saved_at TIMESTAMPTZ
            )
            """
        )
        cur.execute(
            "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS mcp_saved_at TIMESTAMPTZ"
        )
        cur.execute("TRUNCATE bookings, parking_lots RESTART IDENTITY CASCADE")
        cur.execute("INSERT INTO parking_lots (identifier) VALUES ('TEST-LOT-1')")
    return SqlService()


def _create_booking(service, start_hour, end_hour):
    return service.create_booking(
        customer_name="Integration Test",
        car_number=f"TEST{start_hour:04d}",
        start_at=datetime(2030, 1, 1, start_hour, tzinfo=UTC),
        end_at=datetime(2030, 1, 1, end_hour, tzinfo=UTC),
    )


def _claim(service, booking_ids):
    claims = service.claim_pending_bookings()
    return {UUID(str(claim["id"])) for claim in claims} & set(booking_ids)


def test_concurrent_overlapping_approvals_assign_one_lot_once(sql_service):
    first = _create_booking(sql_service, 10, 12)
    second = _create_booking(sql_service, 11, 13)
    booking_ids = {UUID(str(first["id"])), UUID(str(second["id"]))}
    assert _claim(sql_service, booking_ids) == booking_ids

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(sql_service.process_pending_booking, booking_ids))

    assert sorted(result["status"] for result in results) == ["approved", "rejected"]
    approved = [result for result in results if result["status"] == "approved"]
    assert len(approved) == 1
    assert approved[0]["lot_id"] == 1
    with sql_service.target_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM bookings WHERE status = 'approved' AND lot_id = 1"
        )
        assert cur.fetchone()[0] == 1


def test_adjacent_reservations_can_use_the_same_lot(sql_service):
    first = _create_booking(sql_service, 10, 11)
    second = _create_booking(sql_service, 11, 12)
    booking_ids = {UUID(str(first["id"])), UUID(str(second["id"]))}
    assert _claim(sql_service, booking_ids) == booking_ids

    results = [
        sql_service.process_pending_booking(UUID(str(first["id"]))),
        sql_service.process_pending_booking(UUID(str(second["id"]))),
    ]

    assert [result["status"] for result in results] == ["approved", "approved"]
    assert [result["lot_id"] for result in results] == [1, 1]


def test_concurrent_administrator_claims_are_disjoint(sql_service):
    booking = _create_booking(sql_service, 10, 11)
    booking_id = UUID(str(booking["id"]))
    barrier = Barrier(2)

    def claim():
        barrier.wait()
        return _claim(sql_service, {booking_id})

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim(), range(2)))

    assert sorted(len(result) for result in results) == [0, 1]
    assert results[0].isdisjoint(results[1])


def test_expired_claim_is_reclaimed_and_release_clears_claim(sql_service):
    booking = _create_booking(sql_service, 10, 11)
    booking_id = UUID(str(booking["id"]))
    with sql_service.target_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE bookings SET review_claimed_at = now() - interval '31 minutes' WHERE id = %s",
            (booking_id,),
        )

    assert _claim(sql_service, {booking_id}) == {booking_id}
    sql_service.release_booking_claim(booking_id)
    assert _claim(sql_service, {booking_id}) == {booking_id}


def test_processing_rolls_back_on_database_error_and_is_idempotent(sql_service):
    booking = _create_booking(sql_service, 10, 11)
    booking_id = UUID(str(booking["id"]))
    assert _claim(sql_service, {booking_id}) == {booking_id}

    with psycopg.connect(sql_service.get_uri(os.environ["POSTGRES_DB"])) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE OR REPLACE FUNCTION fail_test_booking_update() RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION 'intentional test failure';
                END;
                $$ LANGUAGE plpgsql
                """
            )
            cur.execute(
                """
                CREATE TRIGGER fail_test_booking_update
                BEFORE UPDATE ON bookings
                FOR EACH ROW EXECUTE FUNCTION fail_test_booking_update()
                """
            )

    try:
        with pytest.raises(psycopg.Error, match="intentional test failure"):
            sql_service.process_pending_booking(booking_id)
    finally:
        with psycopg.connect(sql_service.get_uri(os.environ["POSTGRES_DB"])) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DROP TRIGGER IF EXISTS fail_test_booking_update ON bookings"
                )
                cur.execute("DROP FUNCTION IF EXISTS fail_test_booking_update()")

    with sql_service.target_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT status, lot_id FROM bookings WHERE id = %s", (booking_id,))
        assert cur.fetchone() == ("pending", None)

    result = sql_service.process_pending_booking(booking_id)
    repeated = sql_service.process_pending_booking(booking_id)
    assert result["status"] == "approved"
    assert repeated is None
    assert _claim(sql_service, {booking_id}) == {booking_id}
    assert sql_service.mark_reservation_persisted(booking_id)
    assert not sql_service.mark_reservation_persisted(booking_id)
    assert _claim(sql_service, {booking_id}) == set()
