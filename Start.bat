@echo off
cd /d "%~dp0"
chcp 65001 >nul
title brain listen
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-all.ps1" %*
