"""
LangGraph 기반 프로젝트 인원 추천 에이전트
=========================================

이 모듈은 LangGraph를 사용하여 대화형 AI 에이전트를 구현합니다.
사용자의 프로젝트 요청을 분석하고 적합한 인원을 추천하는 전체 워크플로우를 관리합니다.

주요 기능:
- LangGraph 워크플로우 관리
- 사용자 요청 분석 및 구조화
- 참여자 탐색 및 적합도 계산
- 추천 결과 생성 및 포맷팅

작성자: AI Assistant
버전: 2.0.0
"""

from typing import Dict, List, Any, TypedDict, Annotated
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from ..core.data_loader import ProjectDataLoader
from ..core.similarity_analyzer import SimilarityAnalyzer
from ..core.config import OPENAI_API_KEY
from .request_analyzer import RequestAnalyzer
from .participant_analyzer import ParticipantAnalyzer

class AgentState(TypedDict):
    """
    LangGraph 에이전트의 상태를 정의하는 클래스
    
    이 클래스는 에이전트가 처리하는 모든 데이터의 구조를 정의합니다.
    """
    messages: Annotated[List, add_messages]  # 대화 메시지 리스트
    user_request: str                        # 사용자 요청 내용
    recommendations: List[Dict[str, Any]]    # 추천 결과 리스트
    analysis_complete: bool                  # 분석 완료 여부
    analysis: Dict[str, Any]                 # 구조화된 분석 결과


class ProjectRecommendationAgent:
    """
    LangGraph 기반 프로젝트 인원 추천 에이전트
    
    이 클래스는 LangGraph를 사용하여 대화형 AI 에이전트를 구현합니다.
    사용자의 프로젝트 요청을 받아서 적합한 인원을 추천하는 전체 프로세스를 관리합니다.
    
    워크플로우:
    1. 사용자 요청 분석
    2. 적합한 참여자 탐색
    3. 추천 결과 생성
    4. 최종 응답 포맷팅
    """
    
    def __init__(self):
        """
        에이전트 초기화
        
        필요한 컴포넌트들을 초기화하고 LangGraph 워크플로우를 구성합니다.
        """
        # OpenAI GPT 모델 초기화
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # 사용할 GPT 모델
            temperature=0.1,        # 창의성 수준 (낮을수록 일관성 높음)
            api_key=OPENAI_API_KEY
        )
        
        # 핵심 컴포넌트 초기화
        self.data_loader = ProjectDataLoader()
        self.similarity_analyzer = SimilarityAnalyzer()
        
        # 분석기 초기화
        self.request_analyzer = RequestAnalyzer(self.llm)
        self.participant_analyzer = ParticipantAnalyzer(self.data_loader, self.similarity_analyzer)
        
        # LangGraph 워크플로우 구성
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        LangGraph 워크플로우를 구성하는 메서드
        
        Returns:
            StateGraph: 구성된 LangGraph 워크플로우
        """
        workflow = StateGraph(AgentState)
        
        # 워크플로우 노드 추가
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("find_suitable_participants", self._find_suitable_participants)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("format_response", self._format_response)
        
        # 워크플로우 엣지 추가 (순차적 실행)
        workflow.set_entry_point("analyze_request")
        workflow.add_edge("analyze_request", "find_suitable_participants")
        workflow.add_edge("find_suitable_participants", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def _analyze_request(self, state: AgentState) -> AgentState:
        """
        사용자 요청을 분석하는 노드
        
        Args:
            state (AgentState): 현재 에이전트 상태
            
        Returns:
            AgentState: 분석 결과가 추가된 상태
        """
        messages = state["messages"]
        user_message = messages[-1].content if messages else ""
        
        # 요청 분석기 사용
        analysis = self.request_analyzer.analyze_request(user_message)
        
        # 상태 업데이트
        state["user_request"] = user_message
        state["analysis"] = analysis
        state["messages"].append(HumanMessage(content=f"요청 분석 완료: {analysis.get('project_characteristics', '')[:50]}..."))
        
        return state
    
    def _find_suitable_participants(self, state: AgentState) -> AgentState:
        """
        적합한 참여자를 찾는 노드
        
        Args:
            state (AgentState): 현재 에이전트 상태
            
        Returns:
            AgentState: 선별된 참여자들이 추가된 상태
        """
        analysis = state.get("analysis", {})
        
        # 참여자 분석기 사용
        recommendations = self.participant_analyzer.find_suitable_participants(analysis)
        
        # 상태 업데이트
        state["recommendations"] = recommendations
        state["messages"].append(HumanMessage(content=f"참여자 분석 완료: {len(recommendations)}명 선별"))
        
        return state
    
    def _generate_recommendations(self, state: AgentState) -> AgentState:
        """
        추천 결과를 생성하고 정리하는 노드
        
        Args:
            state (AgentState): 현재 에이전트 상태
            
        Returns:
            AgentState: 정리된 추천 결과가 추가된 상태
        """
        recommendations = state["recommendations"]
        
        if not recommendations:
            state["messages"].append(
                HumanMessage(content="요청하신 프로젝트에 적합한 인원을 찾을 수 없습니다.")
            )
            return state
        
        # 상위 5명만 선택하고 상세 정보 정리
        top_recommendations = recommendations[:5]
        detailed_recommendations = []
        
        for rec in top_recommendations:
            detailed_rec = {
                'participant': rec['participant'],
                'total_score': rec['total_score'],
                'recent_score': rec['recent_score'],
                'project_count': rec['project_count'],
                'recent_project_count': rec['recent_project_count'],
                'reasons': rec['reasons'],
                'best_matches': rec['best_matches'][:2]  # 상위 2개 프로젝트만
            }
            detailed_recommendations.append(detailed_rec)
        
        # 상태 업데이트
        state["recommendations"] = detailed_recommendations
        state["analysis_complete"] = True
        state["messages"].append(HumanMessage(content=f"추천 결과 생성 완료: {len(detailed_recommendations)}명"))
        
        return state
    
    def _format_response(self, state: AgentState) -> AgentState:
        """
        최종 응답을 포맷팅하는 노드
        
        Args:
            state (AgentState): 현재 에이전트 상태
            
        Returns:
            AgentState: 포맷팅된 응답이 추가된 상태
        """
        recommendations = state["recommendations"]
        
        if not recommendations:
            response_text = "죄송합니다. 요청하신 프로젝트에 적합한 인원을 찾을 수 없습니다."
        else:
            response_text = "🎯 **프로젝트 인원 추천 결과**\n\n"
            response_text += f"총 {len(recommendations)}명의 적합한 인원을 추천드립니다.\n\n"
            
            for i, rec in enumerate(recommendations, 1):
                response_text += f"## 🏆 {i}. {rec['participant']}\n"
                response_text += f"**적합도 점수**: {rec['total_score']:.2f} | "
                response_text += f"**최근 경험**: {rec['recent_score']:.2f} | "
                response_text += f"**총 프로젝트**: {rec['project_count']}개 | "
                response_text += f"**최근 프로젝트**: {rec['recent_project_count']}개\n\n"
                
                response_text += "### 💡 상세 추천 이유\n"
                for reason in rec['reasons']:
                    response_text += f"• {reason}\n"
                
                if rec['best_matches']:
                    response_text += "\n### 📋 관련 프로젝트 경험\n"
                    for j, match in enumerate(rec['best_matches'], 1):
                        project = match['project']
                        score = match['score']
                        response_text += f"**{j}. {project['프로젝트명']}** (매칭도: {score:.2f})\n"
                        response_text += f"   - 기간: {project['프로젝트기간']}\n"
                        response_text += f"   - 태그: {', '.join(project['프로젝트태그'])}\n"
                        response_text += f"   - 개요: {project['프로젝트개요'][:100]}...\n\n"
                
                response_text += "---\n\n"
        
        # 최종 응답을 상태에 추가
        state["messages"].append(HumanMessage(content=response_text))
        state["messages"].append(HumanMessage(content="추천 완료! 추가 질문이 있으시면 언제든 말씀해 주세요."))
        return state
    
    def process_request(self, user_input: str, timeout_sec: int = 900) -> Dict[str, Any]:
        """
        사용자 요청을 처리하는 메인 메서드
        
        이 메서드는 사용자의 프로젝트 요청을 받아서 적합한 인원을 추천하는
        전체 프로세스를 실행합니다.
        
        Args:
            user_input (str): 사용자의 프로젝트 요청 내용
            
        Returns:
            Dict[str, Any]: 추천 결과와 응답 텍스트를 포함한 딕셔너리
        """
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_request": "",
            "recommendations": [],
            "analysis_complete": False
        }
        
        # LangGraph 워크플로우 실행 (타임아웃 보호)
        def _run_workflow():
            return self.graph.invoke(initial_state)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_workflow)
            try:
                result = future.result(timeout=timeout_sec)
                return {
                    "response": result["messages"][-1].content,
                    "recommendations": result.get("recommendations", [])
                }
            except FuturesTimeoutError:
                return {
                    "response": "⏱️ 처리 시간이 15분을 초과하여 요청을 취소했습니다. 입력을 간소화하거나 다시 시도해 주세요.",
                    "recommendations": []
                }

# 사용 예시
if __name__ == "__main__":
    # 에이전트 초기화
    agent = ProjectRecommendationAgent()
    
    # 테스트 요청 처리
    result = agent.process_request("AI 기반 고객 서비스 개선 프로젝트를 진행하려고 합니다.")
    print(result["response"])
