@echo off
chcp 65001 > nul
echo ========================================================
echo   Запуск обучающего сервиса LA_ML...
echo ========================================================
echo.
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe main.py
) else (
    python main.py
)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ОШИБКА] Не удалось запустить сервис.
    echo Убедитесь, что Python и зависимости установлены.
    pause
)
