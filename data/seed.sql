-- ======================== PARKING LOTS ========================

INSERT INTO parking_lots (id, identifier) VALUES
(1, 'LOT-D01'),
(2, 'LOT-D02'),
(3, 'LOT-D03'),
(4, 'LOT-A01'),
(5, 'LOT-A02'),
(6, 'LOT-A03'),
(7, 'LOT-W01'),
(8, 'LOT-W02'),
(9, 'LOT-U01'),
(10,'LOT-U02'),
(11,'LOT-R01');

-- ======================== PRICES ========================

INSERT INTO prices (price_per_hour, price_per_day, max_stay_hours) VALUES
(6.00, 45.00, 24);

-- ======================== WORKING HOURS ========================

-- --- DEFAULT RECURRING SCHEDULE (day_of_week set, date is NULL) ---

-- Downtown lots: Mon-Sat 6 AM – 11 PM
INSERT INTO working_hours (day_of_week, start_time, end_time) VALUES
(0, '06:00', '23:00'), (1, '06:00', '23:00'), (2, '06:00', '23:00'),
(3, '06:00', '23:00'), (4, '06:00', '23:00'), (5, '06:00', '23:00'),
(6, '06:00', '23:00');

-- --- HOLIDAY / ONE-OFF OVERRIDES (date set, day_of_week is NULL) ---

-- Closed on Thanksgiving
INSERT INTO working_hours (date, start_time, end_time, reason) VALUES
('2026-11-26', '00:00', '00:00', 'Thanksgiving Day — closed');

-- Early closure on Christmas Eve
INSERT INTO working_hours (date, start_time, end_time, reason) VALUES
('2026-12-24', '06:00', '18:00', 'Christmas Eve — early close');

-- ======================== BOOKINGS ========================

INSERT INTO bookings (customer_name, car_number, start_at, end_at, status, lot_id, customer_id) VALUES
('Alice Johnson',   'ABC-1234',  '2026-01-15 08:00', '2026-01-15 14:00', 'approved', 1, 101),
('Bob Smith',       'XYZ-5678',  '2026-01-15 09:30', '2026-01-15 11:00', 'approved', 4, 102),
('Carol Davis',     'DEF-9012',  '2026-01-16 07:00', '2026-01-16 20:00', 'pending',  3, NULL),
('David Lee',       'GHI-3456',  '2026-01-16 10:00', '2026-01-16 12:00', 'rejected', 7, 104),
('Eva Martinez',    'JKL-7890',  '2026-01-17 06:00', '2026-01-17 23:00', 'approved', 5, 105),
('Frank Brown',     'MNO-2345',  '2026-01-17 09:00', '2026-01-17 11:00', 'pending',  9, NULL),
('Grace Kim',       'PQR-6789',  '2026-01-18 08:00', '2026-01-18 16:00', 'approved', 2, 107),
('Hank Wilson',     'STU-0123',  '2026-01-18 14:00', '2026-01-18 15:30', 'pending',  6, NULL),
('Irene Tanaka',    'VWX-4567',  '2026-01-19 07:00', '2026-01-19 19:00', 'approved', 1, 109),
('Jake OBrien',     'YZA-8901',  '2026-01-19 10:00', '2026-01-19 13:00', 'rejected', 8, 110),
('Karen Nguyen',    'BCD-2345',  '2026-01-20 00:00', '2026-01-21 00:00', 'approved', 4, 111),
('Liam Patel',      'EFG-6789',  '2026-01-20 06:00', '2026-01-20 22:00', 'rejected', 3, 112);
