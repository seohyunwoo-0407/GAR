#!/bin/bash

# venv가 없으면 생성
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# .env 파일 로드
set -a
source .env
set +a

echo "Virtual environment activated and environment variables set!"