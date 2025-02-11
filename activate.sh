#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi


set -a
source .env
set +a

echo "Virtual environment activated and environment variables set!"