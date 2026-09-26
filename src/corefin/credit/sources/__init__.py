"""Data source clients for the Credit-Loss Forecasting Engine: FDIC
BankFind (institution identifiers/financials), FRED (macro series), the
Federal Reserve's published supervisory stress test scenarios, and FFIEC
Call Report bulk data. Every client is a thin, typed wrapper around a
network call -- no caching or panel-building logic here (see
credit/panel.py) -- so each is independently testable by mocking the one
HTTP call it makes.
"""
