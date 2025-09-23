"""
유사도 분석 모듈
================

이 모듈은 사용자 요청과 프로젝트 간의 유사도를 분석하고
적합한 인원을 추천하기 위한 핵심 로직을 제공합니다.

주요 기능:
- 텍스트 유사도 계산 (OpenAI Embeddings 사용)
- 프로젝트별 적합도 점수 산출
- 시간 가중치 적용 (최근 프로젝트 우대)
- 상세한 추천 이유 생성

작성자: AI Assistant
버전: 1.1.0
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re
import os
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from .config import VECTOR_DB_DIR, VECTOR_COLLECTION_NAME, PROJECT_DATA_PATH
from .data_loader import ProjectDataLoader
from .config import (
    SIMILARITY_THRESHOLD, 
    RECENT_PROJECT_WEIGHT, 
    TAG_WEIGHT, 
    OVERVIEW_WEIGHT, 
    TITLE_WEIGHT
)

class SimilarityAnalyzer:
    """
    프로젝트와 사용자 요청 간의 유사도를 분석하는 클래스
    
    이 클래스는 한국어 특화 문장 임베딩 모델을 사용하여
    텍스트 간의 의미적 유사도를 계산하고, 다양한 요소를
    종합하여 최종 적합도 점수를 산출합니다.
    """
    
    def __init__(self):
        """
        유사도 분석기 초기화
        
        OpenAI 임베딩을 사용합니다. 환경 변수 `OPENAI_API_KEY`가 설정되어 있어야 합니다.
        """
        self._backend = "openai"
        self._openai_embeddings = None
        self._st_model = None
        self._vector_store = None
        self._metadata_store = None

        # OpenAI embeddings만 사용 (폴백 제거)
        try:
            # text-embedding-3-small: 1536-dim, 빠르고 저렴함
            self._openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self._backend = "openai"
        except Exception as openai_error:
            # OpenAI API 키가 설정되지 않았거나 API 호출 실패
            raise RuntimeError(
                f"OpenAI 임베딩 초기화 실패: {str(openai_error)}\n"
                "OPENAI_API_KEY 환경 변수를 확인하고 유효한 API 키를 설정해주세요."
            ) from openai_error

        # FAISS VectorDB 초기화 (존재하면 로드, 없으면 지연 생성)
        try:
            self._load_faiss_index()
        except Exception:
            self._vector_store = None
            self._metadata_store = None

    # ---------------------------------------------------------------------
    # FAISS 인덱스 관리
    # ---------------------------------------------------------------------
    def _load_faiss_index(self):
        """FAISS 인덱스와 메타데이터를 로드합니다."""
        faiss_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.faiss")
        metadata_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            try:
                self._vector_store = FAISS.load_local(
                    VECTOR_DB_DIR, 
                    self._openai_embeddings,
                    allow_dangerous_deserialization=True
                )
                with open(metadata_path, 'rb') as f:
                    self._metadata_store = pickle.load(f)
                print(f"✅ FAISS 인덱스를 로드했습니다: {faiss_path}")
            except Exception as e:
                print(f"⚠️ FAISS 인덱스 로드 실패: {e}")
                self._vector_store = None
                self._metadata_store = None
        else:
            self._vector_store = None
            self._metadata_store = None

    def _save_faiss_index(self):
        """FAISS 인덱스와 메타데이터를 저장합니다."""
        if self._vector_store is not None and self._metadata_store is not None:
            try:
                # 디렉토리 생성
                os.makedirs(VECTOR_DB_DIR, exist_ok=True)
                
                # FAISS 인덱스 저장
                self._vector_store.save_local(VECTOR_DB_DIR)
                
                # 메타데이터 저장
                metadata_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.pkl")
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self._metadata_store, f)
                
                print(f"✅ FAISS 인덱스를 저장했습니다: {VECTOR_DB_DIR}")
            except Exception as e:
                print(f"❌ FAISS 인덱스 저장 실패: {e}")

    # ---------------------------------------------------------------------
    # 벡터DB 색인/동기화
    # ---------------------------------------------------------------------
    def sync_vector_db(self) -> int:
        """
        프로젝트 데이터를 로드하여 FAISS VectorDB에 임베딩/저장합니다.
        기존 인덱스는 초기화합니다.
        Returns: 저장된 문서 수
        """
        data_loader = ProjectDataLoader()
        projects = data_loader.get_all_projects()
        if not projects:
            return 0

        if self._openai_embeddings is None:
            raise RuntimeError("OpenAI 임베딩이 초기화되지 않았습니다.")

        # 기존 인덱스 파일 삭제
        faiss_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.faiss")
        metadata_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.pkl")
        
        for path in [faiss_path, metadata_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, project in enumerate(projects):
            title = project.get('프로젝트명', '')
            overview = project.get('프로젝트개요', '')
            tags = project.get('프로젝트태그', [])
            joined = f"[제목]{title}\n[개요]{overview}\n[태그]{', '.join(tags)}"
            texts.append(joined)
            metadatas.append({
                "project_name": title,
                "period": project.get('프로젝트기간', ''),
                "tags": project.get('프로젝트태그', []),
                "participants": project.get('참여자명단', []),
                "project_id": f"project_{idx}"
            })

        # FAISS 인덱스 생성
        try:
            self._vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self._openai_embeddings,
                metadatas=metadatas
            )
            self._metadata_store = metadatas
            
            # 인덱스 저장
            self._save_faiss_index()
            
            return len(texts)
        except Exception as e:
            print(f"❌ FAISS 인덱스 생성 실패: {e}")
            return 0

    def search_similar_projects(self, query: str, k: int = 100) -> List[Tuple[Dict[str, Any], float]]:
        """
        FAISS VectorDB에서 질의와 유사한 프로젝트를 상위 k개 반환합니다.
        Returns: [(project_metadata, score), ...]
        """
        if not query.strip():
            return []
        if self._vector_store is None:
            # 연결 시도 (lazy)
            try:
                self._load_faiss_index()
            except Exception:
                return []

        try:
            results = self._vector_store.similarity_search_with_score(query, k=k)
            # results: List[ (Document, score) ]
            formatted: List[Tuple[Dict[str, Any], float]] = []
            for doc, score in results:
                meta = doc.metadata or {}
                # FAISS에서는 거리 점수를 사용하므로 유사도로 변환
                # 거리가 작을수록 유사도가 높으므로 역변환
                distance = float(score)
                # 거리를 0-1 범위의 유사도로 변환 (거리가 0이면 유사도 1, 거리가 클수록 유사도 감소)
                similarity_score = 1.0 / (1.0 + distance)
                formatted.append((meta, similarity_score))
            return formatted
        except Exception as e:
            print(f"❌ FAISS 검색 실패: {e}")
            return []

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        주어진 텍스트 리스트를 임베딩 벡터로 변환합니다.
        OpenAI 임베딩을 사용하여 텍스트를 벡터로 변환합니다.
        """
        if not texts:
            return np.empty((0, 0))

        if self._backend == "openai" and self._openai_embeddings is not None:
            vectors: List[List[float]] = self._openai_embeddings.embed_documents(texts)
            return np.array(vectors, dtype=np.float32)

        # OpenAI 임베딩만 사용하므로 여기에 도달하면 안 됨
        raise RuntimeError("OpenAI 임베딩이 초기화되지 않았습니다.")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 코사인 유사도를 계산하는 메서드
        
        Args:
            text1 (str): 첫 번째 텍스트
            text2 (str): 두 번째 텍스트
            
        Returns:
            float: 유사도 점수 (0.0 ~ 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # 텍스트를 벡터로 변환
        embeddings = self._encode_texts([text1, text2])
        
        # 코사인 유사도 계산
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def calculate_tag_similarity(self, user_tags: List[str], project_tags: List[str]) -> float:
        """
        태그 간 유사도를 계산하는 메서드
        
        Args:
            user_tags (List[str]): 사용자 요청에서 추출한 태그들
            project_tags (List[str]): 프로젝트의 태그들
            
        Returns:
            float: 태그 유사도 점수 (0.0 ~ 1.0)
        """
        if not user_tags or not project_tags:
            return 0.0
        
        # 태그들을 하나의 문자열로 결합하여 유사도 계산
        user_tag_text = " ".join(user_tags)
        project_tag_text = " ".join(project_tags)
        
        return self.calculate_similarity(user_tag_text, project_tag_text)
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 키워드를 추출하는 메서드 (간단한 형태소 분석)
        
        Args:
            text (str): 키워드를 추출할 텍스트
            
        Returns:
            List[str]: 추출된 키워드 리스트
        """
        if not text:
            return []
        
        # 한글, 영문, 숫자만 추출하는 정규표현식
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        
        # 2글자 이상인 단어만 필터링 (의미있는 키워드만 추출)
        keywords = [word for word in words if len(word) >= 2]
        
        return keywords
    
    def calculate_project_similarity(self, user_request: str, project: Dict[str, Any]) -> Dict[str, float]:
        """
        사용자 요청과 프로젝트 간의 유사도를 계산하는 메서드
        
        Args:
            user_request (str): 사용자의 프로젝트 요청 내용
            project (Dict[str, Any]): 비교할 프로젝트 정보
            
        Returns:
            Dict[str, float]: 각 요소별 유사도 점수
        """
        project_title = project.get('프로젝트명', '')
        project_overview = project.get('프로젝트개요', '')
        project_tags = project.get('프로젝트태그', [])
        
        # 각 요소별 유사도 계산
        title_similarity = self.calculate_similarity(user_request, project_title)
        overview_similarity = self.calculate_similarity(user_request, project_overview)
        
        # 태그 유사도 계산
        user_keywords = self.extract_keywords_from_text(user_request)
        tag_similarity = self.calculate_tag_similarity(user_keywords, project_tags)
        
        return {
            'title': title_similarity,
            'overview': overview_similarity,
            'tags': tag_similarity
        }

    def calculate_project_similarity_from_analysis(self, analysis: Dict[str, Any], project: Dict[str, Any]) -> Dict[str, float]:
        """
        Pydantic 분석 결과(특성/태그/필요역량)에 기반한 유사도 계산
        Returns keys: request_similarity, content_similarity, tag_similarity
        """
        if not analysis:
            return {'request_similarity': 0.0, 'content_similarity': 0.0, 'tag_similarity': 0.0}

        required_caps: str = analysis.get('required_capabilities', '') or ''
        project_char: str = analysis.get('project_characteristics', '') or ''
        req_tags: List[str] = analysis.get('tags', []) or []

        project_overview = project.get('프로젝트개요', '') or ''
        project_tags: List[str] = project.get('프로젝트태그', []) or []

        # 1) 필요 역량 vs 프로젝트 개요
        request_similarity = self.calculate_similarity(required_caps, project_overview) if required_caps and project_overview else 0.0

        # 2) 프로젝트 특성 vs 프로젝트 개요
        content_similarity = self.calculate_similarity(project_char, project_overview) if project_char and project_overview else 0.0

        # 3) 태그 리스트 유사도 (요청 태그별로 프로젝트 태그와의 최대 유사도의 평균)
        tag_similarity = 0.0
        if req_tags and project_tags:
            per_tag_scores: List[float] = []
            for t in req_tags:
                # 각 요청 태그 t에 대해 프로젝트 태그들과의 최대 유사도
                if not t:
                    continue
                max_sim = 0.0
                for pt in project_tags:
                    if not pt:
                        continue
                    sim = self.calculate_similarity(str(t), str(pt))
                    if sim > max_sim:
                        max_sim = sim
                per_tag_scores.append(max_sim)
            if per_tag_scores:
                tag_similarity = float(sum(per_tag_scores) / len(per_tag_scores))

        return {
            'request_similarity': float(request_similarity),
            'content_similarity': float(content_similarity),
            'tag_similarity': float(tag_similarity)
        }
    
    def calculate_time_weight(self, project: Dict[str, Any]) -> float:
        """
        프로젝트 수행 시점에 따른 가중치를 계산하는 메서드
        
        최근에 수행한 프로젝트일수록 높은 가중치를 부여합니다.
        
        Args:
            project (Dict[str, Any]): 프로젝트 정보
            
        Returns:
            float: 시간 가중치 (0.8 ~ 1.5)
        """
        try:
            # 프로젝트 종료일 파싱
            end_date_str = project.get('프로젝트기간', '').split(' ~ ')[-1]
            if not end_date_str:
                return 1.0
            
            end_date = datetime.strptime(end_date_str, '%Y.%m')
            current_date = datetime.now()
            
            # 개월 차이 계산
            months_diff = (current_date.year - end_date.year) * 12 + (current_date.month - end_date.month)
            
            # 최근 프로젝트일수록 높은 가중치 적용
            if months_diff <= 6:      # 6개월 이내: 1.5배
                return RECENT_PROJECT_WEIGHT
            elif months_diff <= 12:   # 1년 이내: 1.2배
                return 1.2
            elif months_diff <= 24:   # 2년 이내: 1.0배
                return 1.0
            else:                     # 2년 이상: 0.8배
                return 0.8
                
        except (ValueError, IndexError):
            # 날짜 파싱 실패 시 기본 가중치 적용
            return 1.0
    
    def calculate_participant_suitability(self, analysis: Dict[str, Any], participant_name: str, 
                                        participant_projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        특정 참여자의 적합도를 계산하는 메서드 (FAISS 기반)
        
        이 메서드는 FAISS 벡터 DB를 활용하여 효율적으로 참여자의 적합도를 계산합니다.
        벡터 DB가 없으면 자동으로 동기화를 실행합니다.
        
        Args:
            user_request (str): 사용자의 프로젝트 요청 내용
            participant_name (str): 참여자 이름
            participant_projects (List[Dict[str, Any]]): 참여자의 프로젝트 리스트
            
        Returns:
            Dict[str, Any]: 참여자의 적합도 분석 결과
        """
        if not participant_projects:
            return {
                'participant': participant_name,
                'total_score': 0.0,
                'project_count': 0,
                'recent_score': 0.0,
                'best_matches': [],
                'reasons': []
            }
        
        # FAISS 벡터 DB 확인 및 자동 동기화
        if not self._check_and_sync_vector_db():
            # 벡터 DB 동기화 실패 시 기존 방식으로 폴백
            return self._calculate_participant_suitability_fallback(analysis, participant_name, participant_projects)
        
            # FAISS 기반 유사도 검색 (후보 선별/초기 랭킹용)
        try:
            # 분석 결과를 기반으로 검색 질의 구성
            query_parts: List[str] = []
            req_caps = (analysis or {}).get('required_capabilities') or ''
            proj_char = (analysis or {}).get('project_characteristics') or ''
            tags = (analysis or {}).get('tags') or []
            if req_caps:
                query_parts.append(req_caps)
            if proj_char:
                query_parts.append(proj_char)
            if tags:
                query_parts.append(', '.join(tags))
            query = '\n'.join(query_parts) if query_parts else ''
            similar_projects = self.search_similar_projects(query, k=100) if query else []
            
            # 참여자가 참여한 프로젝트 중에서 매칭되는 것들 필터링
            # 프로젝트명으로 매칭 (더 안정적)
            participant_project_names = {project.get('프로젝트명', '') for project in participant_projects}
            matching_projects = []
            
            for meta, score in similar_projects:
                project_name = meta.get('project_name', '')
                if project_name in participant_project_names:
                    # 원본 프로젝트 데이터 찾기
                    for project in participant_projects:
                        if project.get('프로젝트명', '') == project_name:
                            # 분석 기반 상세 유사도 및 가중치 점수 계산
                            similarities = self.calculate_project_similarity_from_analysis(analysis, project)
                            time_weight = self.calculate_time_weight(project)
                            # 가중치 설정: request 0.5, content 0.25, tags 0.25
                            project_score = (
                                similarities['request_similarity'] * 0.5 +
                                similarities['content_similarity'] * 0.25 +
                                similarities['tag_similarity'] * 0.25
                            ) * time_weight
                            matching_projects.append({
                                'project': project,
                                'score': project_score,
                                'similarities': similarities,
                                'time_weight': time_weight
                            })
                            break
            
            if not matching_projects:
                return {
                    'participant': participant_name,
                    'total_score': 0.0,
                    'project_count': len(participant_projects),
                    'recent_score': 0.0,
                    'recent_project_count': 0,
                    'best_matches': [],
                    'reasons': ["관련 프로젝트 경험이 없습니다."]
                }
            
            # 점수 계산 및 정렬 (폴백과 동일한 방식으로 평균 산출)
            total_score = sum(match['score'] for match in matching_projects)
            avg_score = total_score / len(matching_projects)
            
            # 최근 프로젝트 점수 계산
            recent_score = 0.0
            recent_count = 0
            for match in matching_projects:
                if match['time_weight'] >= RECENT_PROJECT_WEIGHT:
                    recent_score += match['score']
                    recent_count += 1
            
            recent_avg_score = recent_score / recent_count if recent_count > 0 else 0.0
            
            # 상위 매칭 프로젝트들 정렬
            matching_projects.sort(key=lambda x: x['score'], reverse=True)
            best_matches = matching_projects[:3]
            
            # 추천 이유 생성
            reasons = self._generate_recommendation_reasons(participant_name, best_matches, recent_avg_score)
            
            return {
                'participant': participant_name,
                'total_score': avg_score,
                'recent_score': recent_avg_score,
                'project_count': len(participant_projects),
                'recent_project_count': recent_count,
                'best_matches': best_matches,
                'reasons': reasons
            }
            
        except Exception as e:
            print(f"⚠️ FAISS 기반 검색 실패, 폴백 방식 사용: {e}")
            return self._calculate_participant_suitability_fallback(analysis, participant_name, participant_projects)
    
    def _check_and_sync_vector_db(self) -> bool:
        """
        FAISS 벡터 DB가 존재하는지 확인하고, 없으면 자동으로 동기화를 실행합니다.
        
        Returns:
            bool: 벡터 DB가 사용 가능하면 True, 아니면 False
        """
        # 벡터 DB가 이미 로드되어 있는지 확인
        if self._vector_store is not None:
            return True
        
        # 파일 존재 확인
        faiss_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.faiss")
        metadata_path = os.path.join(VECTOR_DB_DIR, f"{VECTOR_COLLECTION_NAME}.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            # 파일이 있으면 로드 시도
            try:
                self._load_faiss_index()
                return self._vector_store is not None
            except Exception as e:
                print(f"⚠️ 기존 벡터 DB 로드 실패: {e}")
                return False
        else:
            # 파일이 없으면 자동 동기화 실행
            print("🔄 FAISS 벡터 DB가 없습니다. 자동으로 동기화를 시작합니다...")
            try:
                doc_count = self.sync_vector_db()
                if doc_count > 0:
                    print(f"✅ 벡터 DB 동기화 완료: {doc_count}개 문서 저장")
                    return True
                else:
                    print("❌ 벡터 DB 동기화 실패")
                    return False
            except Exception as e:
                print(f"❌ 벡터 DB 동기화 중 오류 발생: {e}")
                return False
    
    def _calculate_participant_suitability_fallback(self, analysis: Dict[str, Any], participant_name: str, 
                                                  participant_projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        기존 방식으로 참여자 적합도를 계산하는 폴백 메서드
        
        FAISS가 사용 불가능할 때 기존의 임베딩 방식으로 계산합니다.
        """
        # 1회 한정 재색인 시도 (무한루프 방지 플래그 사용)
        try:
            if not getattr(self, "_reindex_attempted", False):
                self._reindex_attempted = True
                # 인덱스 동기화 시도
                doc_count = self.sync_vector_db()
                if doc_count > 0:
                    # 재색인 성공 시 동일 로직으로 재시도
                    try:
                        result = self.calculate_participant_suitability(analysis, participant_name, participant_projects)
                        return result
                    except Exception:
                        # 재시도 실패 시 폴백 계속 진행
                        pass
        finally:
            # 다음 호출에는 다시 시도할 수 있도록 리셋
            if hasattr(self, "_reindex_attempted"):
                self._reindex_attempted = False

        if not participant_projects:
            return {
                'participant': participant_name,
                'total_score': 0.0,
                'project_count': 0,
                'recent_score': 0.0,
                'best_matches': [],
                'reasons': []
            }
        
        total_score = 0.0
        project_scores = []
        recent_score = 0.0
        recent_count = 0
        
        # 각 프로젝트별로 유사도 계산 (분석 기반)
        for project in participant_projects:
            # 프로젝트별 유사도 계산 (분석 기반)
            similarities = self.calculate_project_similarity_from_analysis(analysis, project)
            
            # 시간 가중치 계산
            time_weight = self.calculate_time_weight(project)
            
            # 종합 점수 계산 (가중 평균)
            project_score = (
                similarities['request_similarity'] * 0.5 +
                similarities['content_similarity'] * 0.25 +
                similarities['tag_similarity'] * 0.25
            ) * time_weight
            
            total_score += project_score
            project_scores.append({
                'project': project,
                'score': project_score,
                'similarities': similarities,
                'time_weight': time_weight
            })
            
            # 최근 프로젝트 점수 (6개월 이내)
            if time_weight >= RECENT_PROJECT_WEIGHT:
                recent_score += project_score
                recent_count += 1
        
        # 평균 점수 계산
        avg_score = total_score / len(participant_projects) if participant_projects else 0.0
        recent_avg_score = recent_score / recent_count if recent_count > 0 else 0.0
        
        # 상위 매칭 프로젝트들 (점수 기준 정렬)
        project_scores.sort(key=lambda x: x['score'], reverse=True)
        best_matches = project_scores[:3]  # 상위 3개
        
        # 추천 이유 생성
        reasons = self._generate_recommendation_reasons(participant_name, best_matches, recent_avg_score)
        
        return {
            'participant': participant_name,
            'total_score': avg_score,
            'recent_score': recent_avg_score,
            'project_count': len(participant_projects),
            'recent_project_count': recent_count,
            'best_matches': best_matches,
            'reasons': reasons
        }

    def _generate_recommendation_reasons(self, participant_name: str, best_matches: List[Dict], 
                                       recent_score: float) -> List[str]:
        """
        추천 이유를 생성하는 내부 메서드
        
        이 메서드는 참여자의 프로젝트 경험을 분석하여
        왜 해당 인원이 적합한지에 대한 구체적인 이유를 생성합니다.
        
        Args:
            participant_name (str): 참여자 이름
            best_matches (List[Dict]): 상위 매칭 프로젝트들
            recent_score (float): 최근 프로젝트 점수
            
        Returns:
            List[str]: 추천 이유 리스트
        """
        reasons = []
        
        if not best_matches:
            return ["관련 프로젝트 경험이 없습니다."]
        
        # 1. 최고 매칭 프로젝트 기반 상세 이유
        best_match = best_matches[0]
        project = best_match['project']
        score = best_match['score']
        similarities = best_match['similarities']
        
        # 프로젝트명 관련성 분석
        if similarities['title'] > 0.6:
            reasons.append(f"🎯 **프로젝트명 유사도 높음**: '{project['프로젝트명']}' 프로젝트와 직접적인 관련성이 있습니다.")
        elif similarities['title'] > 0.4:
            reasons.append(f"📋 **프로젝트명 부분 매칭**: '{project['프로젝트명']}' 프로젝트와 일부 관련성이 있습니다.")
        
        # 프로젝트 개요 관련성 분석
        if similarities['overview'] > 0.7:
            reasons.append(f"📝 **프로젝트 내용 높은 일치**: '{project['프로젝트개요'][:50]}...' 내용과 매우 유사한 경험을 보유합니다.")
        elif similarities['overview'] > 0.5:
            reasons.append(f"📄 **프로젝트 내용 부분 일치**: '{project['프로젝트개요'][:50]}...' 내용과 관련된 경험이 있습니다.")
        
        # 태그 매칭 상세 분석
        if similarities['tags'] > 0.6:
            project_tags = project.get('프로젝트태그', [])
            if project_tags:
                reasons.append(f"🏷️ **전문 분야 일치**: '{', '.join(project_tags[:4])}' 분야에서 전문성을 보유하고 있습니다.")
        elif similarities['tags'] > 0.4:
            project_tags = project.get('프로젝트태그', [])
            if project_tags:
                reasons.append(f"🔖 **관련 분야 경험**: '{', '.join(project_tags[:3])}' 분야와 관련된 경험이 있습니다.")
        
        # 2. 프로젝트 기간 및 최근성 분석
        project_period = project.get('프로젝트기간', '')
        if project_period:
            try:
                end_date_str = project_period.split(' ~ ')[-1]
                if end_date_str:
                    end_date = datetime.strptime(end_date_str, '%Y.%m')
                    current_date = datetime.now()
                    months_diff = (current_date.year - end_date.year) * 12 + (current_date.month - end_date.month)
                    
                    if months_diff <= 6:
                        reasons.append(f"⏰ **최근 경험**: {project_period}에 완료된 최근 프로젝트 경험을 보유합니다.")
                    elif months_diff <= 12:
                        reasons.append(f"📅 **1년 이내 경험**: {project_period}에 완료된 프로젝트 경험을 보유합니다.")
                    else:
                        reasons.append(f"📆 **과거 경험**: {project_period}에 완료된 프로젝트 경험을 보유합니다.")
            except:
                reasons.append(f"📅 **프로젝트 기간**: {project_period}에 참여한 경험이 있습니다.")
        
        # 3. 다중 프로젝트 경험 분석
        if len(best_matches) >= 2:
            second_match = best_matches[1]
            second_project = second_match['project']
            reasons.append(f"🔄 **다양한 관련 경험**: '{second_project['프로젝트명']}' 등 총 {len(best_matches)}개의 관련 프로젝트에 참여했습니다.")
        
        # 4. 최근 활동도 분석
        if recent_score > 0.6:
            reasons.append("🚀 **높은 최근 활동도**: 최근 6개월 내 관련 프로젝트에서 뛰어난 성과를 보였습니다.")
        elif recent_score > 0.4:
            reasons.append("📈 **활발한 최근 활동**: 최근 1년 내 관련 프로젝트에 활발히 참여했습니다.")
        
        # 5. 전체 프로젝트 참여 경험
        total_projects = len([m for m in best_matches if m['score'] > 0.3])
        if total_projects >= 5:
            reasons.append(f"💼 **풍부한 프로젝트 경험**: 총 {total_projects}개 이상의 관련 프로젝트에 참여한 풍부한 경험을 보유합니다.")
        elif total_projects >= 3:
            reasons.append(f"🎯 **충분한 프로젝트 경험**: {total_projects}개의 관련 프로젝트에 참여한 경험이 있습니다.")
        
        # 6. 점수 기반 종합 평가
        if score > 0.8:
            reasons.append("⭐ **매우 높은 적합도**: 요청하신 프로젝트와 매우 높은 관련성을 보입니다.")
        elif score > 0.6:
            reasons.append("✅ **높은 적합도**: 요청하신 프로젝트와 높은 관련성을 보입니다.")
        elif score > 0.4:
            reasons.append("👍 **적절한 적합도**: 요청하신 프로젝트와 적절한 관련성을 보입니다.")
        
        # 7. 특별한 강점이 있는 경우
        if similarities['title'] > 0.7 and similarities['overview'] > 0.7:
            reasons.append("🎖️ **전문성 인증**: 프로젝트명과 내용 모두에서 높은 일치도를 보이는 전문성을 인정받습니다.")
        
        return reasons
