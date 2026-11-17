"""Seed PostgreSQL database with schema and seed data."""

from pathlib import Path

from app.services.sql_service import SqlService

BASE_DIR = Path(__file__).resolve().parents[2]
SCHEMA_FILE = BASE_DIR / "data" / "schema.sql"
SEED_FILE = BASE_DIR / "data" / "seed.sql"


def _read_sql(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text()


def execute_file(conn, path: Path) -> None:
    """Execute all SQL in a file."""
    sql = _read_sql(path)
    with conn.cursor() as cur:
        cur.execute(sql)
    print(f"Executed {path.name}.")


def _print_verification(svc) -> None:
    labels = [
        ("parking_lots", "Parking Lots"),
        ("prices", "Prices"),
        ("working_hours", "Working Hours"),
        ("bookings", "Bookings"),
    ]
    print("\n--- Verification ---")
    with svc.target_conn() as conn, conn.cursor() as cur:
        for table, label in labels:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            print(f"  {label}: {cur.fetchone()[0]} rows")


def seed(svc) -> None:
    """Full reset + schema + seed data pipeline."""
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_FILE}")

    svc.reset_db()

    print(f"\nExecuting schema from {SCHEMA_FILE.name}...")
    with svc.target_conn() as conn:
        execute_file(conn, SCHEMA_FILE)

    print(f"\nExecuting seed data from {SEED_FILE.name}...")
    with svc.target_conn() as conn:
        execute_file(conn, SEED_FILE)

    _print_verification(svc)


def main():
    svc = SqlService()
    seed(svc)
    svc.setup_checkpointer()


if __name__ == "__main__":
    main()
