"""
설정 관리 모듈
==============

이 모듈은 애플리케이션의 모든 설정을 중앙에서 관리합니다.
환경 변수, API 키, 상수값 등을 정의하고 관리합니다.

주요 기능:
- 환경 변수 로드 및 관리
- API 키 설정
- 애플리케이션 상수 정의
- 설정값 검증

작성자: AI Assistant
버전: 1.0.0
"""

import os
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일에서 설정값 읽기)
load_dotenv()

# =============================================================================
# API 설정
# =============================================================================

# OpenAI API 키 설정
# .env 파일에 OPENAI_API_KEY=your_api_key_here 형태로 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# Streamlit 설정
# =============================================================================

# 서버 포트 설정 (기본값: 8501)
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

# 서버 주소 설정 (기본값: localhost)
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

# =============================================================================
# 데이터 경로 설정
# =============================================================================

# 프로젝트 데이터 JSON 파일 경로
PROJECT_DATA_PATH = "./Data/project_info.json"

# 벡터 DB(FAISS) 영속 저장 디렉토리
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "D:/vector_store")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "projects")

# =============================================================================
# 적합도 평가 설정
# =============================================================================

# 유사도 임계값 (이 값 이하는 추천에서 제외)
SIMILARITY_THRESHOLD = 0.3

# 최근 프로젝트 가중치 (6개월 이내 프로젝트에 적용)
RECENT_PROJECT_WEIGHT = 1.5

# 각 요소별 가중치 (총합이 1.0이 되도록 설정)
TAG_WEIGHT = 0.4      # 태그 매칭 가중치
OVERVIEW_WEIGHT = 0.4 # 프로젝트 개요 매칭 가중치  
TITLE_WEIGHT = 0.2    # 프로젝트명 매칭 가중치

# =============================================================================
# 유틸리티 함수
# =============================================================================

def validate_config():
    """
    설정값 검증 함수
    
    Returns:
        bool: 모든 필수 설정이 올바르게 되어 있으면 True, 아니면 False
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        print("⚠️ OpenAI API 키가 설정되지 않았습니다.")
        return False
    
    if not os.path.exists(PROJECT_DATA_PATH):
        print(f"⚠️ 프로젝트 데이터 파일을 찾을 수 없습니다: {PROJECT_DATA_PATH}")
        return False
    
    print("✅ 모든 설정이 올바르게 되어 있습니다.")
    return True

def get_config_summary():
    """
    현재 설정 요약 정보 반환
    
    Returns:
        dict: 설정 정보 딕셔너리
    """
    return {
        "openai_api_configured": bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here"),
        "server_port": STREAMLIT_SERVER_PORT,
        "server_address": STREAMLIT_SERVER_ADDRESS,
        "data_path": PROJECT_DATA_PATH,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "recent_project_weight": RECENT_PROJECT_WEIGHT,
        "weights": {
            "tag": TAG_WEIGHT,
            "overview": OVERVIEW_WEIGHT,
            "title": TITLE_WEIGHT
        }
    }
