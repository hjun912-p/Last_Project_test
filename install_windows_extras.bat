@echo off
REM Windows 전용 추가 패키지 설치
REM requirements.txt 설치 후 실행

echo === Windows 전용 패키지 설치 ===
pip install pywin32==311
pip install pywin32-ctypes==0.2.3
pip install pywinpty==3.0.3
pip install triton-windows==3.6.0.post26

echo === 설치 완료 ===
pause
