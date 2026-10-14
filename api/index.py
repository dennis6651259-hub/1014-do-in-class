"""
api/index.py
Vercel Serverless Function entry point
"""
import os
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel WSGI entrypoint
app = app
