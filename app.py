# app.py — required by some HF Space configurations
# The actual app runs via Dockerfile using uvicorn api.main:app

from api.main import app
