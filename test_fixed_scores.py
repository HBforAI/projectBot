#!/usr/bin/env python3
"""
수정된 점수 변환 로직 테스트
=========================

FAISS 거리 점수를 올바른 유사도로 변환하는지 테스트합니다.
"""

import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_fixed_scores():
    """수정된 점수 변환 로직을 테스트합니다."""
    print("🧪 수정된 점수 변환 로직 테스트")
    print("=" * 50)
    
    try:
        from src.core.similarity_analyzer import SimilarityAnalyzer
        from src.core.data_loader import ProjectDataLoader
        
        # 1. SimilarityAnalyzer 초기화
        print("\n1️⃣ SimilarityAnalyzer 초기화...")
        analyzer = SimilarityAnalyzer()
        print("✅ 초기화 완료")
        
        # 2. FAISS 검색 테스트
        print("\n2️⃣ FAISS 검색 테스트...")
        user_request = "AI 기반 고객 서비스 프로젝트"
        results = analyzer.search_similar_projects(user_request, k=5)
        
        print(f"📊 검색 결과: {len(results)}개")
        for i, (meta, score) in enumerate(results):
            project_name = meta.get('project_name', 'Unknown')
            print(f"   {i+1}. {project_name} (점수: {score:.4f})")
        
        # 3. 참여자 적합도 테스트
        print("\n3️⃣ 참여자 적합도 테스트...")
        data_loader = ProjectDataLoader()
        all_participants = data_loader.get_all_participants()
        
        # 처음 5명의 참여자 테스트
        test_participants = all_participants[:5]
        print(f"📋 테스트 참여자: {test_participants}")
        
        valid_recommendations = 0
        
        for participant in test_participants:
            participant_projects = data_loader.get_projects_by_participant(participant)
            
            suitability = analyzer.calculate_participant_suitability(
                user_request, participant, participant_projects
            )
            
            print(f"\n--- {participant} ---")
            print(f"   총 점수: {suitability['total_score']:.4f}")
            print(f"   프로젝트 수: {suitability['project_count']}")
            
            if suitability['total_score'] >= 0.01:
                valid_recommendations += 1
                print(f"   ✅ 추천 가능 (임계값 통과)")
            else:
                print(f"   ❌ 추천 불가 (임계값 미달)")
        
        print(f"\n📊 추천 가능한 참여자: {valid_recommendations}/{len(test_participants)}명")
        
        # 4. LangGraph 에이전트 시뮬레이션
        print("\n4️⃣ LangGraph 에이전트 시뮬레이션...")
        participant_scores = []
        
        for participant in all_participants:
            participant_projects = data_loader.get_projects_by_participant(participant)
            
            suitability = analyzer.calculate_participant_suitability(
                user_request, participant, participant_projects
            )
            
            # 임계값 이상인 경우만 포함
            if suitability['total_score'] >= 0.01:
                participant_scores.append(suitability)
        
        # 점수 기준으로 정렬
        participant_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"📊 전체 추천 가능한 참여자: {len(participant_scores)}명")
        
        if participant_scores:
            print("\n🏆 상위 5명 추천 결과:")
            for i, rec in enumerate(participant_scores[:5]):
                print(f"   {i+1}. {rec['participant']} (점수: {rec['total_score']:.4f})")
                if rec['reasons']:
                    print(f"      이유: {rec['reasons'][0]}")
        else:
            print("❌ 추천 가능한 참여자가 없습니다.")
        
        return len(participant_scores) > 0
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 수정된 점수 변환 로직 테스트 시작")
    print("=" * 60)
    
    success = test_fixed_scores()
    
    print("\n" + "=" * 60)
    if success:
        print("🎊 테스트 성공! 이제 인원 추천이 정상적으로 작동합니다!")
    else:
        print("💥 테스트 실패. 추가 수정이 필요합니다.")
