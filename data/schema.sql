-- Parking lots
CREATE TABLE IF NOT EXISTS parking_lots (
    id              SERIAL PRIMARY KEY,
    identifier      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Prices
CREATE TABLE IF NOT EXISTS prices (
    id              SERIAL PRIMARY KEY,
    price_per_hour  NUMERIC(6,2) NOT NULL CHECK (price_per_hour >= 0),
    price_per_day   NUMERIC(8,2),           -- optional daily cap
    max_stay_hours  INTEGER,                 -- optional max duration
    effective_from  TIMESTAMPTZ DEFAULT now(),
    effective_to    TIMESTAMPTZ,             -- NULL = still active
    UNIQUE (effective_from)
);

-- Working hours — recurring schedule OR one-off holiday override
CREATE TABLE IF NOT EXISTS working_hours (
    id              SERIAL PRIMARY KEY,
    day_of_week     SMALLINT,                -- 0=Sun … 6=Sat, NULL = holiday row
    start_time      TIME NOT NULL,
    end_time        TIME NOT NULL,
    date            DATE,                    -- for holiday / one-off rows
    reason          TEXT,                    -- e.g. "New Year's Day"

    CHECK (day_of_week IS NULL OR (day_of_week >= 0 AND day_of_week <= 6)),
    CHECK ((day_of_week IS NOT NULL) OR (date IS NOT NULL))  -- one must be filled
);

-- Bookings
CREATE TABLE IF NOT EXISTS bookings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_name   TEXT NOT NULL,
    car_number      TEXT NOT NULL,
    start_at        TIMESTAMPTZ NOT NULL,
    end_at          TIMESTAMPTZ NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','approved','rejected','cancelled')),
    lot_id          INTEGER REFERENCES parking_lots(id) ON DELETE SET NULL,
    customer_id     INTEGER,                 -- nullable = guest / walk-in
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    review_claimed_at TIMESTAMPTZ,
    mcp_saved_at    TIMESTAMPTZ
);

ALTER TABLE bookings
    ADD COLUMN IF NOT EXISTS review_claimed_at TIMESTAMPTZ;

ALTER TABLE bookings
    ADD COLUMN IF NOT EXISTS mcp_saved_at TIMESTAMPTZ;
