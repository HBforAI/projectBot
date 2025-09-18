#!/usr/bin/env python3
"""
프로젝트 인원 추천 챗봇 메인 실행 파일
====================================

이 파일은 프로젝트 인원 추천 챗봇의 메인 진입점입니다.
Streamlit 웹 애플리케이션을 실행합니다.

사용법:
    python main.py

작성자: AI Assistant
버전: 1.0.0
"""

import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """
    필요한 패키지 설치 확인
    
    Returns:
        bool: 모든 패키지가 설치되어 있으면 True
    """
    try:
        import streamlit
        import langchain
        import langgraph
        import sentence_transformers
        import plotly
        import pandas
        import numpy
        import sklearn
        print("✅ 모든 필요한 패키지가 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
        print("다음 명령어로 패키지를 설치해주세요:")
        print("pip install -r requirements.txt")
        return False

def check_env_file():
    """
    환경 변수 파일 확인
    
    Returns:
        bool: 환경 변수가 올바르게 설정되어 있으면 True
    """
    env_file = Path("./.env")
    if not env_file.exists():
        print("⚠️ .env 파일이 없습니다.")
        print("config.py 파일을 참고하여 .env 파일을 생성해주세요.")
        return False
    
    # .env 파일에서 OPENAI_API_KEY 확인
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        if "OPENAI_API_KEY" not in content or "your_openai_api_key_here" in content:
            print("⚠️ .env 파일에 올바른 OPENAI_API_KEY가 설정되지 않았습니다.")
            return False
    
    print("✅ 환경 변수 파일이 올바르게 설정되어 있습니다.")
    return True

def main():
    """
    메인 실행 함수
    """
    print("🚀 프로젝트 인원 추천 챗봇을 시작합니다...")
    
    # 현재 디렉토리를 스크립트가 있는 디렉토리로 변경
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 요구사항 확인
    if not check_requirements():
        sys.exit(1)
    
    # 환경 변수 확인
    if not check_env_file():
        print("\n환경 변수 설정 후 다시 실행해주세요.")
        sys.exit(1)
    
    # Streamlit 앱 실행
    print("\n🌐 Streamlit 앱을 시작합니다...")
    print("브라우저에서 http://localhost:8501 을 열어주세요.")
    print("종료하려면 Ctrl+C를 누르세요.\n")
    
    try:
        # Streamlit 앱 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 챗봇을 종료합니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
