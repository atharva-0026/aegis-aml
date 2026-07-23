"""
Small pure utility functions used by the Streamlit frontend.

Kept separate from app.py so they can be unit tested without
needing a running Streamlit context.
"""


def clean_nav(page_str):
    """Strip emojis and leading/trailing whitespace from a nav label
    to get its canonical ASCII form, e.g. ' Single Scan' -> 'Single Scan'."""
    return page_str.encode('ascii', 'ignore').decode('ascii').strip()
