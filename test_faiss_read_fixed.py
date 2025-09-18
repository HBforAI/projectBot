#!/usr/bin/env python3
"""
FAISS 저장된 데이터 읽기 테스트 스크립트 (수정된 버전)
=================================================

실제 저장된 FAISS 파일을 올바르게 읽어오는 방법을 보여줍니다.
"""

import os
import sys
import pickle
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_faiss_data_reading_fixed():
    """실제 저장된 FAISS 데이터를 읽어오는 방법을 보여줍니다."""
    print("🔍 FAISS 저장된 데이터 읽기 테스트 (수정된 버전)")
    print("=" * 60)
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
        from src.core.config import VECTOR_DB_DIR
        
        print(f"📂 벡터 DB 디렉토리: {VECTOR_DB_DIR}")
        
        # 실제 저장된 파일 확인
        faiss_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
        pkl_path = os.path.join(VECTOR_DB_DIR, "index.pkl")
        
        print(f"📁 FAISS 파일: {faiss_path}")
        print(f"📁 PKL 파일: {pkl_path}")
        
        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            print("✅ 저장된 파일들이 존재합니다.")
            
            # 임베딩 모델 초기화
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # FAISS 인덱스 로드
            print("🔄 FAISS 인덱스를 로드하는 중...")
            vector_store = FAISS.load_local(
                VECTOR_DB_DIR, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS 인덱스 로드 성공")
            
            # 벡터 검색 테스트
            print("\n🔍 벡터 검색 테스트:")
            results = vector_store.similarity_search_with_score("AI 프로젝트", k=5)
            print(f"   검색 결과: {len(results)}개")
            
            for i, (doc, score) in enumerate(results):
                print(f"\n   {i+1}. 거리 점수: {score:.4f}")
                print(f"      내용: {doc.page_content[:100]}...")
                print(f"      메타데이터: {doc.metadata}")
            
            # 인덱스 정보 확인
            print(f"\n📊 인덱스 정보:")
            print(f"   총 벡터 수: {vector_store.index.ntotal}")
            print(f"   벡터 차원: {vector_store.index.d}")
            
        else:
            print("❌ 저장된 파일이 없습니다.")
            print(f"   FAISS 파일 존재: {os.path.exists(faiss_path)}")
            print(f"   PKL 파일 존재: {os.path.exists(pkl_path)}")
            
    except Exception as e:
        print(f"❌ FAISS 읽기 실패: {e}")
        import traceback
        traceback.print_exc()

def test_metadata_reading():
    """메타데이터만 읽어오는 방법을 보여줍니다."""
    print("\n" + "=" * 60)
    print("📋 메타데이터 읽기 테스트")
    print("=" * 60)
    
    try:
        from src.core.config import VECTOR_DB_DIR
        
        pkl_path = os.path.join(VECTOR_DB_DIR, "index.pkl")
        
        if os.path.exists(pkl_path):
            print(f"📁 메타데이터 파일: {pkl_path}")
            
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"✅ 메타데이터 로드 성공: {len(metadata)}개 항목")
            
            # 메타데이터 샘플 출력
            print("\n📋 메타데이터 샘플 (처음 3개):")
            for i, meta in enumerate(metadata[:3]):
                print(f"\n   {i+1}. 프로젝트명: {meta.get('project_name', 'Unknown')}")
                print(f"      기간: {meta.get('period', 'Unknown')}")
                print(f"      태그: {meta.get('tags', [])}")
                print(f"      참여자: {meta.get('participants', [])}")
                print(f"      프로젝트 ID: {meta.get('project_id', 'Unknown')}")
        else:
            print(f"❌ 메타데이터 파일이 없습니다: {pkl_path}")
            
    except Exception as e:
        print(f"❌ 메타데이터 읽기 실패: {e}")
        import traceback
        traceback.print_exc()

def show_usage_examples():
    """실제 사용 예제를 보여줍니다."""
    print("\n" + "=" * 60)
    print("📚 FAISS 데이터 읽기 실제 사용법")
    print("=" * 60)
    
    print("""
🔧 방법 1: SimilarityAnalyzer 사용 (가장 간단)
--------------------------------------------
from src.core.similarity_analyzer import SimilarityAnalyzer

analyzer = SimilarityAnalyzer()
results = analyzer.search_similar_projects("AI 프로젝트", k=5)

for meta, score in results:
    print(f"프로젝트: {meta['project_name']}, 점수: {score:.4f}")

🔧 방법 2: 직접 FAISS 로드 (고급 사용)
------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# FAISS 인덱스 로드
vector_store = FAISS.load_local(
    "D:/vector_store", 
    embeddings,
    allow_dangerous_deserialization=True
)

# 검색 실행
results = vector_store.similarity_search_with_score("검색어", k=5)
for doc, score in results:
    print(f"내용: {doc.page_content}")
    print(f"메타데이터: {doc.metadata}")
    print(f"거리: {score}")

🔧 방법 3: 메타데이터만 읽기
--------------------------
import pickle

with open("D:/vector_store/index.pkl", 'rb') as f:
    metadata = pickle.load(f)

print(f"총 {len(metadata)}개의 프로젝트 데이터")
for meta in metadata:
    print(f"프로젝트: {meta['project_name']}")
    print(f"태그: {meta['tags']}")
    """)

if __name__ == "__main__":
    test_faiss_data_reading_fixed()
    test_metadata_reading()
    show_usage_examples()
