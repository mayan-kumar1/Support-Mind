import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# Read API URL from Streamlit secrets
try:
    os.environ["API_BASE_URL"] = st.secrets["API_BASE_URL"]
except Exception:
    pass

from ui.app import *
