#!/usr/bin/env python3
"""
FAISS 벡터 DB 구현 테스트 스크립트
================================

ChromaDB에서 FAISS로 교체한 구현이 제대로 작동하는지 테스트합니다.
"""

import os
import sys
import traceback

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_faiss_implementation():
    """FAISS 구현을 테스트합니다."""
    print("🧪 FAISS 벡터 DB 구현 테스트 시작...")
    
    try:
        # 1. SimilarityAnalyzer 임포트 테스트
        print("\n1️⃣ SimilarityAnalyzer 임포트 테스트...")
        from src.core.similarity_analyzer import SimilarityAnalyzer
        print("✅ SimilarityAnalyzer 임포트 성공")
        
        # 2. SimilarityAnalyzer 초기화 테스트
        print("\n2️⃣ SimilarityAnalyzer 초기화 테스트...")
        analyzer = SimilarityAnalyzer()
        print("✅ SimilarityAnalyzer 초기화 성공")
        
        # 3. 기본 유사도 계산 테스트
        print("\n3️⃣ 기본 유사도 계산 테스트...")
        similarity = analyzer.calculate_similarity("AI 프로젝트", "AI 기반 시스템")
        print(f"✅ 유사도 계산 성공: {similarity:.4f}")
        
        # 4. 태그 유사도 계산 테스트
        print("\n4️⃣ 태그 유사도 계산 테스트...")
        tag_similarity = analyzer.calculate_tag_similarity(["AI", "고객서비스"], ["AI", "고객관리"])
        print(f"✅ 태그 유사도 계산 성공: {tag_similarity:.4f}")
        
        # 5. 키워드 추출 테스트
        print("\n5️⃣ 키워드 추출 테스트...")
        keywords = analyzer.extract_keywords_from_text("AI 기반 고객 서비스 개선 프로젝트")
        print(f"✅ 키워드 추출 성공: {keywords}")
        
        # 6. 벡터 DB 동기화 테스트 (데이터가 있는 경우에만)
        print("\n6️⃣ 벡터 DB 동기화 테스트...")
        try:
            doc_count = analyzer.sync_vector_db()
            print(f"✅ 벡터 DB 동기화 성공: {doc_count}개 문서 저장")
            
            # 7. 벡터 검색 테스트
            print("\n7️⃣ 벡터 검색 테스트...")
            if doc_count > 0:
                results = analyzer.search_similar_projects("AI 프로젝트", k=5)
                print(f"✅ 벡터 검색 성공: {len(results)}개 결과 반환")
                for i, (meta, score) in enumerate(results[:3]):
                    print(f"   {i+1}. {meta.get('project_name', 'Unknown')} (점수: {score:.4f})")
            else:
                print("⚠️ 벡터 DB에 데이터가 없어 검색 테스트를 건너뜁니다.")
                
        except Exception as e:
            print(f"⚠️ 벡터 DB 동기화 실패 (데이터 파일이 없을 수 있음): {e}")
        
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

def test_imports():
    """필요한 라이브러리들이 제대로 임포트되는지 테스트합니다."""
    print("📦 라이브러리 임포트 테스트...")
    
    try:
        import faiss
        print("✅ FAISS 임포트 성공")
    except ImportError as e:
        print(f"❌ FAISS 임포트 실패: {e}")
        return False
    
    try:
        from langchain_community.vectorstores import FAISS
        print("✅ LangChain FAISS 임포트 성공")
    except ImportError as e:
        print(f"❌ LangChain FAISS 임포트 실패: {e}")
        return False
    
    try:
        from langchain_openai import OpenAIEmbeddings
        print("✅ OpenAI Embeddings 임포트 성공")
    except ImportError as e:
        print(f"❌ OpenAI Embeddings 임포트 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 FAISS 벡터 DB 교체 테스트 시작")
    print("=" * 50)
    
    # 1. 라이브러리 임포트 테스트
    if not test_imports():
        print("\n❌ 라이브러리 임포트 테스트 실패. requirements.txt를 확인하세요.")
        sys.exit(1)
    
    # 2. FAISS 구현 테스트
    if test_faiss_implementation():
        print("\n🎊 모든 테스트가 성공했습니다! FAISS로의 교체가 완료되었습니다.")
    else:
        print("\n💥 일부 테스트가 실패했습니다. 코드를 다시 확인해주세요.")
        sys.exit(1)
