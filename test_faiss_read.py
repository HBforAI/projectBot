#!/usr/bin/env python3
"""
FAISS 저장된 데이터 읽기 테스트 스크립트
=====================================

FAISS에 저장된 벡터 데이터와 메타데이터를 읽어오는 방법을 보여줍니다.
"""

import os
import sys
import pickle
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_faiss_data_reading():
    """FAISS에 저장된 데이터를 읽어오는 방법을 보여줍니다."""
    print("🔍 FAISS 저장된 데이터 읽기 테스트")
    print("=" * 50)
    
    # 1. SimilarityAnalyzer를 통한 읽기 (권장 방법)
    print("\n1️⃣ SimilarityAnalyzer를 통한 읽기 (권장 방법)")
    try:
        from src.core.similarity_analyzer import SimilarityAnalyzer
        from src.core.config import VECTOR_DB_DIR, VECTOR_COLLECTION_NAME
        
        analyzer = SimilarityAnalyzer()
        
        # 벡터 검색 테스트
        print("📊 벡터 검색 테스트:")
        results = analyzer.search_similar_projects("AI 프로젝트", k=3)
        print(f"   검색 결과: {len(results)}개")
        for i, (meta, score) in enumerate(results):
            print(f"   {i+1}. {meta.get('project_name', 'Unknown')} (점수: {score:.4f})")
            
    except Exception as e:
        print(f"❌ SimilarityAnalyzer 읽기 실패: {e}")
    
    # 2. 직접 파일에서 읽기
    print("\n2️⃣ 직접 파일에서 읽기")
    try:
        from src.core.config import VECTOR_DB_DIR, VECTOR_COLLECTION_NAME
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        
        faiss_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.faiss")
        metadata_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.pkl")
        
        print(f"📁 FAISS 파일 경로: {faiss_path}")
        print(f"📁 메타데이터 파일 경로: {metadata_path}")
        
        # 파일 존재 확인
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            print("✅ 저장된 파일들이 존재합니다.")
            
            # FAISS 인덱스 로드
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.load_local(
                VECTOR_DB_DIR, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS 인덱스 로드 성공")
            
            # 메타데이터 로드
            with open(metadata_path, 'rb') as f:
                metadata_store = pickle.load(f)
            print(f"✅ 메타데이터 로드 성공: {len(metadata_store)}개 항목")
            
            # 메타데이터 샘플 출력
            print("\n📋 메타데이터 샘플 (처음 3개):")
            for i, meta in enumerate(metadata_store[:3]):
                print(f"   {i+1}. {meta}")
            
            # 벡터 검색 테스트
            print("\n🔍 벡터 검색 테스트:")
            results = vector_store.similarity_search_with_score("AI 프로젝트", k=3)
            for i, (doc, score) in enumerate(results):
                print(f"   {i+1}. {doc.page_content[:50]}... (거리: {score:.4f})")
                print(f"      메타데이터: {doc.metadata}")
            
        else:
            print("❌ 저장된 파일이 없습니다.")
            print("   먼저 sync_vector_db()를 실행해서 데이터를 저장하세요.")
            
    except Exception as e:
        print(f"❌ 직접 읽기 실패: {e}")
    
    # 3. 저장된 파일 정보 확인
    print("\n3️⃣ 저장된 파일 정보 확인")
    try:
        from src.core.config import VECTOR_DB_DIR, VECTOR_COLLECTION_NAME
        
        faiss_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.faiss")
        metadata_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.pkl")
        
        print(f"📂 벡터 DB 디렉토리: {VECTOR_DB_DIR}")
        print(f"📂 컬렉션 이름: {VECTOR_COLLECTION_NAME}")
        
        if os.path.exists(VECTOR_DB_DIR):
            print(f"✅ 디렉토리 존재: {VECTOR_DB_DIR}")
            files = os.listdir(VECTOR_DB_DIR)
            print(f"📁 디렉토리 내 파일들: {files}")
        else:
            print(f"❌ 디렉토리가 존재하지 않습니다: {VECTOR_DB_DIR}")
            
    except Exception as e:
        print(f"❌ 파일 정보 확인 실패: {e}")

def show_faiss_usage_examples():
    """FAISS 사용 예제를 보여줍니다."""
    print("\n" + "=" * 60)
    print("📚 FAISS 데이터 읽기 사용법")
    print("=" * 60)
    
    print("""
🔧 방법 1: SimilarityAnalyzer 사용 (권장)
----------------------------------------
from src.core.similarity_analyzer import SimilarityAnalyzer

analyzer = SimilarityAnalyzer()
results = analyzer.search_similar_projects("검색어", k=5)

🔧 방법 2: 직접 FAISS 로드
-------------------------
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pickle

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# FAISS 인덱스 로드
vector_store = FAISS.load_local(
    "D:/vector_store", 
    embeddings,
    allow_dangerous_deserialization=True
)

# 메타데이터 로드
with open("D:/vector_store/projects.pkl", 'rb') as f:
    metadata = pickle.load(f)

# 검색 실행
results = vector_store.similarity_search_with_score("검색어", k=5)

🔧 방법 3: 메타데이터만 읽기
--------------------------
import pickle

with open("D:/vector_store/projects.pkl", 'rb') as f:
    metadata = pickle.load(f)

print(f"총 {len(metadata)}개의 프로젝트 데이터")
for i, meta in enumerate(metadata[:3]):
    print(f"{i+1}. {meta['project_name']}")
    """)

if __name__ == "__main__":
    test_faiss_data_reading()
    show_faiss_usage_examples()
