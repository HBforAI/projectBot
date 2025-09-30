"""
요청 분석 모듈
==============

사용자 요청 분석 및 검증을 담당하는 모듈입니다.

주요 기능:
- 사용자 요청을 구조화된 데이터로 변환
- Pydantic 검증 및 재시도 로직
- 폴백 분석 결과 생성

작성자: AI Assistant
버전: 1.0.0
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class RequestAnalysis(BaseModel):
    """사용자 요청 분석 결과 (구조화)"""
    project_characteristics: str = Field(
        min_length=1,
        description="프로젝트의 목적과 분야, 그리고 도메인은 무엇인지에 대한 설명"
    )
    tags: List[str] = Field(
        default_factory=list,
        min_length=1,
        description="프로젝트의 도메인/분야를 나타내는 핵심 키워드 기반의 태그 목록"
    )
    required_capabilities: str = Field(
        min_length=1,
        description="프로젝트 수행을 위해 필요해 보이는 역량에 대한 설명"
    )
    
    @field_validator('project_characteristics', 'required_capabilities')
    @classmethod
    def validate_string_not_empty(cls, v):
        """문자열 필드가 비어있지 않은지 검증"""
        if not v or not v.strip():
            raise ValueError('이 필드는 비어있을 수 없습니다.')
        return v.strip()
    
    @field_validator('tags')
    @classmethod
    def validate_tags_not_empty(cls, v):
        """태그 리스트가 비어있지 않은지 검증"""
        if not v or len(v) == 0:
            raise ValueError('태그는 최소 1개 이상이어야 합니다.')
        # 빈 문자열이나 공백만 있는 태그 제거
        filtered_tags = [tag.strip() for tag in v if tag and tag.strip()]
        if not filtered_tags:
            raise ValueError('유효한 태그가 최소 1개 이상이어야 합니다.')
        return filtered_tags


class RequestAnalyzer:
    """
    사용자 요청 분석을 담당하는 클래스
    
    이 클래스는 사용자 요청을 구조화된 데이터로 변환하고 검증합니다.
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        요청 분석기 초기화
        
        Args:
            llm: OpenAI LLM 모델
        """
        self.llm = llm
    
    def analyze_request(self, user_message: str) -> Dict[str, Any]:
        """
        사용자 요청을 분석하여 구조화된 데이터로 변환
        
        null 값이 나오면 최대 3번까지 재시도합니다.
        
        Args:
            user_message: 사용자 요청 메시지
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # Pydantic 파서 준비
        parser = PydanticOutputParser(pydantic_object=RequestAnalysis)
        format_instructions = parser.get_format_instructions()

        # 프롬프트 구성 (구조화 출력 지시 포함)
        prompt = PromptTemplate(
            template=(
                "다음 사용자 요청을 분석하여 주어진 스키마에 맞는 JSON을 생성하세요.\n"
                "- project_characteristics: 프로젝트의 목적/분야/도메인 설명 (필수, 비어있으면 안됨)\n"
                "- tags: 프로젝트 도메인/분야 태그 목록 (예: HR, 거시경제, 바이오) (필수, 최소 1개 이상)\n"
                "- required_capabilities: 수행에 필요한 역량 설명 (필수, 비어있으면 안됨)\n\n"
                "⚠️ 중요: 모든 필드는 반드시 유효한 값을 가져야 합니다. 빈 값이나 null은 허용되지 않습니다.\n\n"
                "반드시 아래 형식 지침을 따르세요:\n{format_instructions}\n\n"
                "사용자 요청:\n{user_request}"
            ),
            input_variables=["user_request"],
            partial_variables={"format_instructions": format_instructions},
        )

        # 최대 3번까지 재시도
        max_retries = 3
        analysis: Dict[str, Any] = None
        
        for attempt in range(max_retries):
            try:
                # LLM 호출
                raw = self.llm.invoke([
                    SystemMessage(content="당신은 프로젝트 요청을 분석하여 구조화된 결과를 생성하는 전문가입니다. 모든 필드는 반드시 유효한 값을 가져야 합니다."),
                    HumanMessage(content=prompt.format(user_request=user_message))
                ])

                # 파싱 시도
                parsed = parser.parse(raw.content)
                analysis = parsed.model_dump()
                
                # null 값 체크
                if self._has_null_values(analysis):
                    print(f"⚠️ 시도 {attempt + 1}/{max_retries}: null 값 발견, 재시도 중...")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        print("❌ 최대 재시도 횟수 초과, 기본값으로 폴백")
                        analysis = self._get_fallback_analysis(user_message)
                else:
                    print(f"✅ 분석 성공 (시도 {attempt + 1}/{max_retries})")
                    break
                    
            except ValidationError as e:
                print(f"⚠️ 시도 {attempt + 1}/{max_retries}: 검증 실패 - {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    print("❌ 최대 재시도 횟수 초과, 기본값으로 폴백")
                    analysis = self._get_fallback_analysis(user_message)
                    break

        return analysis
    
    def _has_null_values(self, analysis: Dict[str, Any]) -> bool:
        """
        분석 결과에 null 값이 있는지 확인
        
        Args:
            analysis: 분석 결과 딕셔너리
            
        Returns:
            bool: null 값이 있으면 True, 없으면 False
        """
        if not analysis:
            return True
            
        # project_characteristics 체크
        if not analysis.get('project_characteristics') or not analysis.get('project_characteristics').strip():
            return True
            
        # required_capabilities 체크
        if not analysis.get('required_capabilities') or not analysis.get('required_capabilities').strip():
            return True
            
        # tags 체크
        tags = analysis.get('tags', [])
        if not tags or len(tags) == 0:
            return True
            
        # 태그 중에 빈 값이 있는지 체크
        valid_tags = [tag for tag in tags if tag and tag.strip()]
        if not valid_tags:
            return True
            
        return False
    
    def _get_fallback_analysis(self, user_message: str) -> Dict[str, Any]:
        """
        재시도 실패 시 사용할 기본 분석 결과
        
        Args:
            user_message: 사용자 요청 메시지
            
        Returns:
            Dict[str, Any]: 기본 분석 결과
        """
        # 사용자 메시지에서 키워드 추출 시도
        keywords = []
        if "AI" in user_message or "인공지능" in user_message:
            keywords.append("AI")
        if "데이터" in user_message:
            keywords.append("데이터분석")
        if "웹" in user_message or "웹사이트" in user_message:
            keywords.append("웹개발")
        if "앱" in user_message or "모바일" in user_message:
            keywords.append("모바일앱")
        if "머신러닝" in user_message or "ML" in user_message:
            keywords.append("머신러닝")
        
        # 기본 키워드가 없으면 일반적인 키워드 사용
        if not keywords:
            keywords = ["프로젝트관리", "기획"]
        
        return {
            "project_characteristics": user_message[:100] + "..." if len(user_message) > 100 else user_message,
            "tags": keywords,
            "required_capabilities": "프로젝트 수행을 위한 기본 역량"
        }
