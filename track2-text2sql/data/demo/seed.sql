INSERT INTO finance_demo.customers (customer_id, customer_name, city) VALUES
    (1, 'Alice', 'Shanghai'),
    (2, 'Bob', 'Beijing'),
    (3, 'Carol', 'Shanghai');

INSERT INTO finance_demo.loans
    (loan_id, customer_id, product_type, status, amount, issued_at)
VALUES
    (101, 1, 'consumer', 'approved', 12000.00, '2025-01-10 09:00:00+08'),
    (102, 2, 'auto', 'pending', 80000.00, '2025-01-31 23:30:00+08'),
    (103, 3, 'consumer', 'approved', 36000.00, '2025-02-01 00:30:00+08');

