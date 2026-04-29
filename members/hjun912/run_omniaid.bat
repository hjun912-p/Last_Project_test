@echo off
echo OmniAID Web Service is starting...
cd OmniAID

:: Miniconda 환경 활성화
call C:\Users\june4\miniconda3\Scripts\activate.bat LP

:: Python 실행
python app.py

:: 에러 체크
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Web Service failed to start with error code %errorlevel%.
    pause
)
pause
