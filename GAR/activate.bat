@echo off

:: 절대 경로로 .venv 지정
set VENV_PATH=C:\Users\shw41\GAR\.venv

:: .venv 폴더 사용
if exist "%VENV_PATH%" (
    echo Found existing .venv at %VENV_PATH%
    call %VENV_PATH%\Scripts\activate.bat
) else (
    echo Virtual environment .venv not found!
    echo Creating virtual environment...
    python -m venv %VENV_PATH%
    call %VENV_PATH%\Scripts\activate.bat
    pip install -r requirements.txt
)

:: 환경변수 설정
set PYTHONPATH=%PYTHONPATH%;%CD%
set VIRTUAL_ENV=%VENV_PATH%
set PATH=%VIRTUAL_ENV%\Scripts;%PATH%

echo Virtual environment activated and environment variables set! 