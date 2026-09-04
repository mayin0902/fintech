CREATE SCHEMA IF NOT EXISTS finance_demo;

CREATE TABLE finance_demo.customers (
    customer_id BIGINT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    city TEXT NOT NULL
);

CREATE TABLE finance_demo.loans (
    loan_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL REFERENCES finance_demo.customers(customer_id),
    product_type TEXT NOT NULL,
    status TEXT NOT NULL,
    amount NUMERIC(14, 2) NOT NULL,
    issued_at TIMESTAMPTZ NOT NULL
);

