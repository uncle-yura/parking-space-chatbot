import os
from contextlib import asynccontextmanager, contextmanager
from uuid import UUID

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import sql


class SqlService:
    """Manages PostgreSQL admin lifecycle + parking data queries."""

    @staticmethod
    def _admin_params() -> dict:
        return {
            "host": os.environ.get("POSTGRES_HOST", "localhost"),
            "port": int(os.environ.get("POSTGRES_PORT", "5432")),
            "user": os.environ.get("POSTGRES_USER", "postgres"),
            "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
        }

    @classmethod
    def get_uri(cls, dbname) -> str:
        params = cls._admin_params()
        return (
            f"postgres://{params['user']}:{params['password']}"
            f"@{params['host']}:{params['port']}/{dbname}?sslmode=disable"
        )

    def get_db_params(self) -> dict:
        admin = self._admin_params()
        admin["dbname"] = os.environ.get("POSTGRES_DB", "parking_chatbot")
        return admin

    @contextmanager
    def _connect(self, dbname: str):
        """Yield an autocommit connection."""
        p = self._admin_params()
        conn = psycopg.connect(**p, dbname=dbname)
        conn.autocommit = True
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def admin_conn(self):
        """Yield an autocommit connection to the 'postgres' admin DB."""
        with self._connect("postgres") as conn:
            yield conn

    @contextmanager
    def target_conn(self):
        """Yield an autocommit connection to the parking database."""
        with self._connect(self.get_db_params()["dbname"]) as conn:
            yield conn

    @contextmanager
    def checkpointer(self):
        DB_URI = self.get_uri("postgres")
        with PostgresSaver.from_conn_string(DB_URI) as cp:
            yield cp

    @asynccontextmanager
    async def async_checkpointer(self):
        """Yield an async PostgreSQL checkpointer for async LangGraph runs."""
        db_uri = self.get_uri("postgres")
        async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
            yield checkpointer

    def drop_db(self) -> None:
        """Drop the parking database if it exists."""
        db = self.get_db_params()["dbname"]
        with self.admin_conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db,))
            if not cur.fetchone():
                print(f"Database '{db}' does not exist — nothing to drop.")
                return

            cur.execute(
                """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid();
                    """,
                (db,),
            )
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(db)))
            print(f"Dropped database '{db}'.")

    def ensure_db(self) -> None:
        """Create the parking database if it doesn't exist."""
        db = self.get_db_params()["dbname"]
        with self.admin_conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db,))
            if cur.fetchone():
                print(f"Database '{db}' already exists.")
                return
            cur.execute(
                sql.SQL("CREATE DATABASE {} TEMPLATE template0;").format(
                    sql.Identifier(db)
                )
            )
            print(f"Created database '{db}'.")

    def reset_db(self) -> None:
        """Drop then recreate — idempotent full wipe."""
        self.drop_db()
        self.ensure_db()

    def setup_checkpointer(self) -> None:
        DB_URI = self.get_uri("postgres")
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()

    # ------------------------------------------------------------------
    # Queries (for AI tools)
    # ------------------------------------------------------------------

    def get_free_parking_lots(self, start_at=None, end_at=None) -> list[dict]:
        """Count spaces without an approved booking overlapping the requested period."""
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT COUNT(*) AS free_lots
                    FROM parking_lots pl
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM bookings b
                        WHERE b.lot_id = pl.id
                          AND b.status = 'approved'
                          AND b.start_at < COALESCE(CAST(%s AS timestamptz), now() + interval '1 second')
                          AND b.end_at > COALESCE(CAST(%s AS timestamptz), now())
                    );
                    """,
                (end_at, start_at),
            )
            cols = [d.name for d in cur.description or []]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_prices(self, as_of=None) -> list[dict]:
        """Return the most recent price effective on the requested date/time."""
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT p.price_per_hour, p.price_per_day,
                           p.max_stay_hours, p.effective_from
                    FROM prices p
                    WHERE p.effective_from <= COALESCE(CAST(%s AS timestamptz), now())
                      AND (p.effective_to IS NULL OR p.effective_to > COALESCE(CAST(%s AS timestamptz), now()))
                    ORDER BY p.effective_from DESC
                    LIMIT 1;
                    """,
                (as_of, as_of),
            )
            cols = [d.name for d in cur.description or []]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_working_hours(self) -> list[dict]:
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT wh.day_of_week, wh.start_time, wh.end_time, wh.date, wh.reason
                    FROM working_hours wh
                    ORDER BY COALESCE(wh.date, '0001-01-01'), wh.day_of_week;
                    """
            )
            cols = [d.name for d in cur.description or []]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_booking_status(self, booking_id: UUID) -> dict | None:
        """Return the status and assigned lot for one booking, without customer PII."""
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT b.id, b.status, pl.identifier AS lot_identifier
                FROM bookings b
                LEFT JOIN parking_lots pl ON pl.id = b.lot_id
                WHERE b.id = %s;
                """,
                (booking_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d.name for d in cur.description or []]
            return dict(zip(cols, row))

    def claim_pending_bookings(self) -> list[dict]:
        """Atomically claim pending requests so concurrent admins get disjoint work."""
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH eligible AS (
                    SELECT id
                    FROM bookings
                    WHERE (
                          status = 'pending'
                          OR (status = 'approved' AND mcp_saved_at IS NULL)
                      )
                      AND (
                          review_claimed_at IS NULL
                          OR review_claimed_at < now() - interval '30 minutes'
                      )
                    ORDER BY created_at ASC, id ASC
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE bookings AS b
                SET review_claimed_at = now()
                FROM eligible AS e
                WHERE b.id = e.id
                RETURNING b.id, b.customer_name, b.car_number, b.start_at,
                          b.end_at, b.status, b.created_at;
                """
            )
            columns = [description.name for description in cur.description or []]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def release_booking_claim(self, booking_id: UUID) -> None:
        """Release an unprocessed request so another admin run can retry it."""
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bookings SET review_claimed_at = NULL
                WHERE id = %s AND status IN ('pending', 'approved');
                """,
                (booking_id,),
            )

    def mark_reservation_persisted(self, booking_id: UUID) -> bool:
        """Record successful MCP persistence and clear its processing claim."""
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bookings
                SET mcp_saved_at = now(), review_claimed_at = NULL
                WHERE id = %s AND status = 'approved' AND mcp_saved_at IS NULL
                RETURNING id;
                """,
                (booking_id,),
            )
            return cur.fetchone() is not None

    def reject_pending_booking(self, booking_id: UUID) -> dict | None:
        """Reject a pending reservation after agent review."""
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bookings
                SET status = 'rejected', review_claimed_at = NULL
                WHERE id = %s AND status = 'pending'
                  AND review_claimed_at IS NOT NULL
                RETURNING id;
                """,
                (booking_id,),
            )
            return (
                {"status": "rejected", "lot_id": None, "lot_identifier": None}
                if cur.fetchone()
                else None
            )

    def process_pending_booking(self, booking_id: UUID) -> dict | None:
        """Approve a pending request if a lot is free, otherwise reject it.

        Availability selection and status update are serialized in one transaction,
        so concurrent administrator runs cannot assign the same lot for overlapping
        reservations.
        """
        with self.target_conn() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_xact_lock(681204, 1)")
                    cur.execute(
                        """
                        SELECT start_at, end_at
                        FROM bookings
                        WHERE id = %s AND status = 'pending'
                          AND review_claimed_at IS NOT NULL
                        FOR UPDATE;
                        """,
                        (booking_id,),
                    )
                    booking = cur.fetchone()
                    if booking is None:
                        conn.commit()
                        return None

                    start_at, end_at = booking
                    cur.execute(
                        """
                        SELECT pl.id, pl.identifier
                        FROM parking_lots pl
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM bookings b
                            WHERE b.lot_id = pl.id
                              AND b.status = 'approved'
                              AND b.start_at < %s
                              AND b.end_at > %s
                        )
                        ORDER BY pl.id
                        LIMIT 1;
                        """,
                        (end_at, start_at),
                    )
                    lot = cur.fetchone()
                    status = "approved" if lot else "rejected"
                    lot_id = lot[0] if lot else None
                    cur.execute(
                        """
                        UPDATE bookings
                        SET status = %s, lot_id = %s, review_claimed_at = NULL
                        WHERE id = %s AND status = 'pending'
                          AND review_claimed_at IS NOT NULL
                        RETURNING id;
                        """,
                        (status, lot_id, booking_id),
                    )
                    updated = cur.fetchone() is not None
                conn.commit()
                return (
                    {"status": status, "lot_id": lot_id, "lot_identifier": lot[1]}
                    if updated and lot
                    else {"status": status, "lot_id": None, "lot_identifier": None}
                    if updated
                    else None
                )
            except Exception:
                conn.rollback()
                raise

    def create_booking(
        self,
        customer_name: str,
        car_number: str,
        start_at,
        end_at,
        lot_id: int | None = None,
        customer_id: int | None = None,
    ) -> dict:
        """Insert a booking and return the row with generated id."""
        sql_str = """
            INSERT INTO bookings (customer_name, car_number, start_at, end_at, lot_id, customer_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, status, created_at;
        """
        with self.target_conn() as conn, conn.cursor() as cur:
            cur.execute(
                sql_str,
                (
                    customer_name.strip(),
                    car_number.strip().upper(),
                    start_at,
                    end_at,
                    lot_id,
                    customer_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Booking insert did not return a row.")
            cols = [description.name for description in cur.description or []]
            return dict(zip(cols, row))
