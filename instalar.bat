@echo off
echo === Instalando CodificaMed ===
python -m venv .venv || (echo No se pudo crear el entorno virtual. Verifique que Python 3.11 o superior este instalado. & exit /b 1)
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Fallo la instalacion de librerias. & exit /b 1)
if not exist .env copy .env.example .env
echo.
echo === Listo ===
echo Siguiente paso: ollama pull qwen2.5:3b y ollama pull bge-m3
