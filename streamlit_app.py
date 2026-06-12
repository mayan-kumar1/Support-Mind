import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set API URL from secrets before importing app
import streamlit as st

try:
    os.environ["API_BASE_URL"] = st.secrets["API_BASE_URL"]
except Exception:
    pass

# Import only the UI — not the agent
exec(open(os.path.join(os.path.dirname(__file__), "ui", "app.py")).read())
