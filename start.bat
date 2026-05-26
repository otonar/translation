@echo off
chcp 65001 > nul
echo ===================================================
echo  リアルタイム英語→日本語翻訳アプリ セットアップ
echo ===================================================
echo.
echo [1/2] 依存パッケージをインストール中...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] インストールに失敗しました。
    echo pip が使えることを確認してください: python -m pip --version
    pause
    exit /b 1
)
echo.
echo [2/2] アプリを起動します...
echo.
python main.py
pause
