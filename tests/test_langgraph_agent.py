#!/usr/bin/env python3
"""
LangGraph 에이전트 테스트 스크립트
================================

실제 LangGraph 에이전트가 제대로 작동하는지 테스트합니다.
"""

import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_langgraph_agent():
    """LangGraph 에이전트를 테스트합니다."""
    print("🧪 LangGraph 에이전트 테스트")
    print("=" * 50)
    
    try:
        from src.agents.langgraph_agent import ProjectRecommendationAgent
        
        # 1. 에이전트 초기화
        print("\n1️⃣ ProjectRecommendationAgent 초기화...")
        agent = ProjectRecommendationAgent()
        print("✅ 초기화 완료")
        
        # 2. 사용자 요청 처리 테스트
        print("\n2️⃣ 사용자 요청 처리 테스트...")
        user_input = "AI 기반 고객 서비스 프로젝트"
        print(f"📝 사용자 요청: '{user_input}'")
        
        # 3. 요청 처리 실행
        print("\n🔄 요청 처리 중...")
        result = agent.process_request(user_input, timeout_sec=60)
        
        # 4. 결과 확인
        print("\n📊 처리 결과:")
        print(f"   분석 완료: {result.get('analysis_complete', False)}")
        print(f"   추천 수: {len(result.get('recommendations', []))}")
        
        if result.get('recommendations'):
            print("\n🏆 추천 결과:")
            for i, rec in enumerate(result['recommendations'][:3]):
                print(f"   {i+1}. {rec['participant']} (점수: {rec['total_score']:.4f})")
                if rec['reasons']:
                    print(f"      이유: {rec['reasons'][0]}")
        else:
            print("❌ 추천 결과가 없습니다.")
        
        # 5. 응답 텍스트 확인
        print(f"\n📝 응답 텍스트 (처음 200자):")
        response_text = result.get('response_text', '')
        print(f"   {response_text[:200]}...")
        
        return len(result.get('recommendations', [])) > 0
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_nodes():
    """개별 노드들을 테스트합니다."""
    print("\n" + "=" * 50)
    print("🔍 개별 노드 테스트")
    print("=" * 50)
    
    try:
        from src.agents.langgraph_agent import ProjectRecommendationAgent
        from langchain.schema import HumanMessage
        
        agent = ProjectRecommendationAgent()
        
        # 1. analyze_request 노드 테스트
        print("\n1️⃣ analyze_request 노드 테스트...")
        state = {
            "messages": [HumanMessage(content="AI 기반 고객 서비스 프로젝트")],
            "user_request": "",
            "recommendations": [],
            "analysis_complete": False
        }
        
        result_state = agent._analyze_request(state)
        print(f"   사용자 요청: {result_state['user_request']}")
        print(f"   메시지 수: {len(result_state['messages'])}")
        
        # 2. find_suitable_participants 노드 테스트
        print("\n2️⃣ find_suitable_participants 노드 테스트...")
        result_state = agent._find_suitable_participants(result_state)
        print(f"   추천 수: {len(result_state['recommendations'])}")
        
        if result_state['recommendations']:
            print("   상위 3명:")
            for i, rec in enumerate(result_state['recommendations'][:3]):
                print(f"      {i+1}. {rec['participant']} (점수: {rec['total_score']:.4f})")
        else:
            print("   ❌ 추천 결과가 없습니다.")
        
        # 3. generate_recommendations 노드 테스트
        print("\n3️⃣ generate_recommendations 노드 테스트...")
        result_state = agent._generate_recommendations(result_state)
        print(f"   최종 추천 수: {len(result_state['recommendations'])}")
        
        return len(result_state['recommendations']) > 0
        
    except Exception as e:
        print(f"❌ 개별 노드 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 LangGraph 에이전트 테스트 시작")
    print("=" * 60)
    
    # 1. 개별 노드 테스트
    success1 = test_individual_nodes()
    
    # 2. 전체 에이전트 테스트
    success2 = test_langgraph_agent()
    
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    print(f"✅ 개별 노드 테스트: {'성공' if success1 else '실패'}")
    print(f"✅ 전체 에이전트 테스트: {'성공' if success2 else '실패'}")
    
    if success1 and success2:
        print("\n🎊 모든 테스트가 성공했습니다!")
    else:
        print("\n💥 일부 테스트가 실패했습니다.")
        print("🔧 LangGraph 에이전트 로직을 확인해주세요.")
