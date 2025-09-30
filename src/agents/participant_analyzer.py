"""
참여자 분석 모듈
================

참여자 관련 분석 및 필터링 로직을 담당하는 모듈입니다.

주요 기능:
- 참여자별 프로젝트 매핑 생성
- FAISS 검색 결과 기반 참여자 선별
- 참여자 적합도 계산 및 정렬

작성자: AI Assistant
버전: 1.0.0
"""

from typing import Dict, List, Any, Tuple
import os
from ..core.data_loader import ProjectDataLoader
from ..core.similarity_analyzer import SimilarityAnalyzer
from ..core.config import VECTOR_DB_DIR, VECTOR_COLLECTION_NAME


class ParticipantAnalyzer:
    """
    참여자 분석 및 필터링을 담당하는 클래스
    
    이 클래스는 참여자 관련 모든 분석 로직을 처리합니다.
    """
    
    def __init__(self, data_loader: ProjectDataLoader, similarity_analyzer: SimilarityAnalyzer):
        """
        참여자 분석기 초기화
        
        Args:
            data_loader: 프로젝트 데이터 로더
            similarity_analyzer: 유사도 분석기
        """
        self.data_loader = data_loader
        self.similarity_analyzer = similarity_analyzer
    
    def find_suitable_participants(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        적합한 참여자를 찾는 메인 메서드 (최적화된 버전)
        
        FAISS 검색을 한 번만 수행하고, 매칭되는 참여자만 선별하여 성능을 최적화합니다.
        
        Args:
            analysis: 사용자 요청 분석 결과
            
        Returns:
            List[Dict[str, Any]]: 선별된 참여자 목록
        """
        all_participants = self.data_loader.get_all_participants()
        
        print(f"🔍 총 {len(all_participants)}명의 참여자 중에서 적합한 인원을 찾는 중...")
        
        # 1. FAISS 벡터 DB 확인 및 자동 동기화
        if not self._check_and_sync_vector_db():
            print("❌ FAISS 벡터 DB를 사용할 수 없습니다. 폴백 방식으로 진행합니다.")
            return self._find_suitable_participants_fallback(analysis)
        
        # 2. FAISS 검색을 한 번만 수행 (가장 비용이 큰 작업)
        print("📊 FAISS 검색 수행 중...")
        similar_projects = self.similarity_analyzer.search_similar_projects(
            self._build_search_query(analysis), k=30
        )
        print(f"✅ FAISS 검색 완료: {len(similar_projects)}개 프로젝트 발견")
        
        # 2. 참여자별 프로젝트 매핑을 미리 생성
        participant_project_map = self._build_participant_project_map()
        
        # 3. 매칭되는 참여자와 관련 프로젝트만 선별 (1차 필터링)
        matching_participants_with_projects = self._find_matching_participants_with_projects(
            similar_projects, participant_project_map
        )
        print(f"🎯 1차 필터링 완료: {len(matching_participants_with_projects)}명의 후보 참여자 선별")
        
        participant_scores = []
        
        # 4. 선별된 참여자만 상세 계산
        for i, (participant, matching_projects) in enumerate(matching_participants_with_projects.items()):
            print(f"📈 참여자 {i+1}/{len(matching_participants_with_projects)}: {participant} 분석 중...")
            print(f"   📋 매칭된 프로젝트: {len(matching_projects)}개")
            
            # 적합도 계산 (SimilarityAnalyzer 내부 캐시 사용)
            suitability = self.similarity_analyzer.calculate_participant_suitability(
                analysis, participant, matching_projects
            )
            
            # 임계값 이상인 경우만 포함
            if suitability['total_score'] >= 0.01:
                participant_scores.append(suitability)
        
        # 점수 기준으로 정렬
        participant_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"✅ 최종 추천: {len(participant_scores)}명의 적합한 참여자 선별 완료")
        
        # 상위 10명만 선택
        return participant_scores[:10]
    
    def _build_search_query(self, analysis: Dict[str, Any]) -> str:
        """
        분석 결과를 기반으로 검색 쿼리 구성
        
        Args:
            analysis: 사용자 요청 분석 결과
            
        Returns:
            str: 검색 쿼리
        """
        query_parts = []
        req_caps = analysis.get('required_capabilities', '') or ''
        proj_char = analysis.get('project_characteristics', '') or ''
        tags = analysis.get('tags', []) or []
        
        if req_caps:
            query_parts.append(req_caps)
        if proj_char:
            query_parts.append(proj_char)
        if tags:
            query_parts.append(', '.join(tags))
        
        return '\n'.join(query_parts) if query_parts else ''
    
    def _build_participant_project_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        참여자별 프로젝트 매핑을 미리 생성
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: 참여자별 프로젝트 목록
        """
        all_participants = self.data_loader.get_all_participants()
        participant_project_map = {}
        
        for participant in all_participants:
            participant_projects = self.data_loader.get_projects_by_participant(participant)
            participant_project_map[participant] = participant_projects
        
        return participant_project_map
    
    def _find_matching_participants_with_projects(self, similar_projects: List[Tuple[Dict[str, Any], float]], 
                                                participant_project_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        FAISS 검색 결과에서 매칭되는 참여자들과 관련 프로젝트들을 선별
        
        Args:
            similar_projects: FAISS 검색 결과
            participant_project_map: 참여자별 프로젝트 매핑
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 참여자별 매칭된 프로젝트 목록
        """
        matching_participants_with_projects = {}
        
        # FAISS 검색 결과에서 프로젝트명 추출
        similar_project_names = set()
        for meta, score in similar_projects:
            project_name = meta.get('project_name', '')
            if project_name:
                similar_project_names.add(project_name)
        
        # 각 참여자에 대해 매칭된 프로젝트만 선별
        for participant, all_projects in participant_project_map.items():
            matching_projects = []
            for project in all_projects:
                project_name = project.get('프로젝트명', '')
                if project_name in similar_project_names:
                    matching_projects.append(project)
            
            # 매칭된 프로젝트가 있는 참여자만 포함
            if matching_projects:
                matching_participants_with_projects[participant] = matching_projects
        
        return matching_participants_with_projects
    
    def _check_and_sync_vector_db(self) -> bool:
        """
        FAISS 벡터 DB가 존재하는지 확인하고, 없으면 자동으로 동기화를 실행합니다.
        
        Returns:
            bool: 벡터 DB가 사용 가능하면 True, 아니면 False
        """
        # SimilarityAnalyzer의 벡터 DB 상태 확인
        if hasattr(self.similarity_analyzer, '_vector_store') and self.similarity_analyzer._vector_store is not None:
            return True
        
        # 파일 존재 확인
        faiss_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.faiss")
        metadata_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            # 파일이 있으면 SimilarityAnalyzer에서 로드 시도
            try:
                if hasattr(self.similarity_analyzer, '_load_faiss_index'):
                    self.similarity_analyzer._load_faiss_index()
                    return hasattr(self.similarity_analyzer, '_vector_store') and self.similarity_analyzer._vector_store is not None
                return False
            except Exception as e:
                print(f"⚠️ 기존 벡터 DB 로드 실패: {e}")
                return False
        else:
            # 파일이 없으면 자동 동기화 실행
            print("🔄 FAISS 벡터 DB가 없습니다. 자동으로 동기화를 시작합니다...")
            try:
                if hasattr(self.similarity_analyzer, 'sync_vector_db'):
                    doc_count = self.similarity_analyzer.sync_vector_db()
                    if doc_count > 0:
                        print(f"✅ 벡터 DB 동기화 완료: {doc_count}개 문서 저장")
                        return True
                    else:
                        print("❌ 벡터 DB 동기화 실패")
                        return False
                return False
            except Exception as e:
                print(f"❌ 벡터 DB 동기화 중 오류 발생: {e}")
                return False
    
    def _find_suitable_participants_fallback(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        FAISS를 사용할 수 없을 때 기존 방식으로 참여자를 찾는 폴백 메서드
        
        Args:
            analysis: 사용자 요청 분석 결과
            
        Returns:
            List[Dict[str, Any]]: 선별된 참여자 목록
        """
        print("🔄 폴백 방식으로 참여자 분석을 진행합니다...")
        
        all_participants = self.data_loader.get_all_participants()
        participant_scores = []
        
        # 모든 참여자에 대해 기존 방식으로 계산
        for i, participant in enumerate(all_participants):
            print(f"📈 참여자 {i+1}/{len(all_participants)}: {participant} 분석 중...")
            
            # 참여자의 프로젝트들 가져오기
            participant_projects = self.data_loader.get_projects_by_participant(participant)
            
            # 적합도 계산 (폴백 방식)
            suitability = self.similarity_analyzer._calculate_participant_suitability_fallback(
                analysis, participant, participant_projects
            )
            
            # 임계값 이상인 경우만 포함
            if suitability['total_score'] >= 0.01:
                participant_scores.append(suitability)
        
        # 점수 기준으로 정렬
        participant_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"✅ 폴백 방식 완료: {len(participant_scores)}명의 적합한 참여자 선별")
        
        # 상위 10명만 선택
        return participant_scores[:10]
