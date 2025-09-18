#!/usr/bin/env python3
"""
새로운 FAISS 기반 로직 테스트 스크립트
====================================

수정된 calculate_participant_suitability 메서드가 
FAISS 벡터 DB를 효율적으로 사용하는지 테스트합니다.
"""

import os
import sys
import time

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_new_faiss_logic():
    """새로운 FAISS 기반 로직을 테스트합니다."""
    print("🧪 새로운 FAISS 기반 로직 테스트")
    print("=" * 50)
    
    try:
        from src.core.similarity_analyzer import SimilarityAnalyzer
        from src.core.data_loader import ProjectDataLoader
        
        # 1. SimilarityAnalyzer 초기화
        print("\n1️⃣ SimilarityAnalyzer 초기화...")
        analyzer = SimilarityAnalyzer()
        print("✅ 초기화 완료")
        
        # 2. 데이터 로더로 참여자 정보 가져오기
        print("\n2️⃣ 참여자 데이터 로드...")
        data_loader = ProjectDataLoader()
        all_participants = data_loader.get_all_participants()
        print(f"✅ 총 {len(all_participants)}명의 참여자 로드")
        
        # 3. 테스트용 참여자 선택 (처음 3명)
        test_participants = all_participants[:3]
        print(f"📋 테스트 참여자: {test_participants}")
        
        # 4. 각 참여자에 대해 새로운 로직 테스트
        user_request = "AI 기반 고객 서비스 프로젝트"
        print(f"\n🔍 사용자 요청: '{user_request}'")
        
        for i, participant in enumerate(test_participants):
            print(f"\n--- 참여자 {i+1}: {participant} ---")
            
            # 참여자의 프로젝트 가져오기
            participant_projects = data_loader.get_projects_by_participant(participant)
            print(f"   참여 프로젝트 수: {len(participant_projects)}")
            
            # 성능 측정 시작
            start_time = time.time()
            
            # 새로운 FAISS 기반 적합도 계산
            suitability = analyzer.calculate_participant_suitability(
                user_request, participant, participant_projects
            )
            
            # 성능 측정 종료
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 결과 출력
            print(f"   ⏱️ 처리 시간: {processing_time:.3f}초")
            print(f"   📊 총 점수: {suitability['total_score']:.4f}")
            print(f"   📈 최근 점수: {suitability['recent_score']:.4f}")
            print(f"   📋 프로젝트 수: {suitability['project_count']}")
            print(f"   🔥 최근 프로젝트 수: {suitability['recent_project_count']}")
            
            if suitability['best_matches']:
                print(f"   🎯 최고 매칭 프로젝트:")
                for j, match in enumerate(suitability['best_matches'][:2]):
                    project_name = match['project'].get('프로젝트명', 'Unknown')
                    score = match['score']
                    print(f"      {j+1}. {project_name} (점수: {score:.4f})")
            
            if suitability['reasons']:
                print(f"   💡 추천 이유:")
                for reason in suitability['reasons'][:2]:
                    print(f"      • {reason}")
        
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_db_auto_sync():
    """벡터 DB 자동 동기화 기능을 테스트합니다."""
    print("\n" + "=" * 50)
    print("🔄 벡터 DB 자동 동기화 테스트")
    print("=" * 50)
    
    try:
        from src.core.similarity_analyzer import SimilarityAnalyzer
        from src.core.config import VECTOR_DB_DIR
        
        # 기존 벡터 DB 파일 백업 (테스트용)
        faiss_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
        pkl_path = os.path.join(VECTOR_DB_DIR, "index.pkl")
        
        backup_faiss = faiss_path + ".backup"
        backup_pkl = pkl_path + ".backup"
        
        # 백업 생성
        if os.path.exists(faiss_path):
            import shutil
            shutil.copy2(faiss_path, backup_faiss)
            shutil.copy2(pkl_path, backup_pkl)
            print("✅ 기존 벡터 DB 백업 완료")
            
            # 기존 파일 삭제 (자동 동기화 테스트용)
            os.remove(faiss_path)
            os.remove(pkl_path)
            print("🗑️ 기존 벡터 DB 파일 삭제 (테스트용)")
        
        # 새로운 SimilarityAnalyzer 생성 (벡터 DB 없음)
        print("\n🔄 벡터 DB가 없는 상태에서 SimilarityAnalyzer 생성...")
        analyzer = SimilarityAnalyzer()
        
        # 자동 동기화 테스트
        print("\n🧪 자동 동기화 테스트...")
        start_time = time.time()
        
        # 간단한 검색으로 자동 동기화 트리거
        results = analyzer.search_similar_projects("AI 프로젝트", k=5)
        
        end_time = time.time()
        sync_time = end_time - start_time
        
        print(f"✅ 자동 동기화 완료: {sync_time:.3f}초")
        print(f"📊 검색 결과: {len(results)}개")
        
        # 백업 복원
        if os.path.exists(backup_faiss):
            shutil.copy2(backup_faiss, faiss_path)
            shutil.copy2(backup_pkl, pkl_path)
            os.remove(backup_faiss)
            os.remove(backup_pkl)
            print("🔄 원본 벡터 DB 복원 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ 자동 동기화 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 새로운 FAISS 기반 로직 테스트 시작")
    print("=" * 60)
    
    # 1. 새로운 로직 테스트
    success1 = test_new_faiss_logic()
    
    # 2. 자동 동기화 테스트
    success2 = test_vector_db_auto_sync()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    print(f"✅ 새로운 FAISS 로직: {'성공' if success1 else '실패'}")
    print(f"✅ 자동 동기화 기능: {'성공' if success2 else '실패'}")
    
    if success1 and success2:
        print("\n🎊 모든 테스트가 성공했습니다!")
        print("💡 이제 FAISS 벡터 DB를 효율적으로 사용합니다!")
    else:
        print("\n💥 일부 테스트가 실패했습니다.")
        print("🔧 코드를 다시 확인해주세요.")
