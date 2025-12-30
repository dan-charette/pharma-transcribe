#!/bin/bash

# PharmaTranscribe AI - Start Script

cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py
