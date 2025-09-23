"""
Streamlit을 기반으로 하는 프로젝트 인원 추천 챗봇의 사용자 인터페이스 모듈입니다.
사용자 입력을 받고, LangGraph 에이전트와 연동하여 추천 결과를 시각적으로 표시합니다.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
import os
import sys
from datetime import datetime

# 프로젝트 루트 디렉토리를 sys.path에 추가하여 모듈 임포트 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

from src.core.data_loader import ProjectDataLoader
from src.agents.langgraph_agent import ProjectRecommendationAgent
from src.core.config import OPENAI_API_KEY, PROJECT_DATA_PATH
from src.core.similarity_analyzer import SimilarityAnalyzer

# --- Streamlit 앱 설정 ---
st.set_page_config(
    page_title="프로젝트 인원 추천 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 스타일링 ---
st.markdown("""
<style>
.main-header {
    font-size: 2.5em;
    font-weight: bold;
    color: #007bff;
    text-align: center;
    margin-bottom: 1.5em;
    border-bottom: 2px solid #007bff;
    padding-bottom: 0.5em;
}
.stButton>button {
    background-color: #007bff;
    color: white;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-size: 1.1em;
    font-weight: bold;
    border: none;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #0056b3;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.recommendation-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-left: 8px solid #28a745;
}
.recommendation-card h3 {
    color: #28a745;
    margin-top: 0;
    margin-bottom: 0.5rem;
}
.score-badge {
    display: inline-block;
    background-color: #007bff;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-size: 0.85rem;
    font-weight: bold;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}
.reason-item {
    background-color: #f8f9fa;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    font-size: 0.9rem;
    line-height: 1.4;
    color: #000 !important;
}
.reason-item a, .reason-item strong, .reason-item em, .reason-item span { color: #000 !important; }
.project-item {
    background-color: #ffffff;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 5px;
    border: 1px solid #e9ecef;
    font-size: 0.85rem;
}
.project-item strong {
    color: #495057;
}
</style>
""", unsafe_allow_html=True)

# --- 전역 변수 및 캐싱 ---
@st.cache_resource
def load_data() -> ProjectDataLoader:
    """ProjectDataLoader 인스턴스를 로드하고 캐싱합니다."""
    return ProjectDataLoader()

@st.cache_resource
def load_agent() -> ProjectRecommendationAgent:
    """ProjectRecommendationAgent 인스턴스를 로드하고 캐싱합니다."""
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API 키가 설정되지 않았습니다. `.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
        return None
    return ProjectRecommendationAgent()

# --- 사이드바 메뉴 구성 ---
def render_sidebar():
    """사이드바에 계층적 메뉴를 구성하고 선택된 메뉴를 반환합니다."""
    st.sidebar.markdown("## 🎯 메뉴")
    
    # session_state 초기화
    if 'selected_menu' not in st.session_state:
        st.session_state.selected_menu = "recommendation"
    
    # Tree Menu 구조 정의 (확장 가능)
    menu_structure = {
        "🚀 인원 추천": {
            "value": "recommendation",
            "children": {}
        },
        "📊 프로젝트 통계": {
            "value": "statistics", 
            "children": {}
        },
        "📈 데이터 분석": {
            "value": "analysis",
            "children": {}
        },
        "❓ 도움말": {
            "value": "help",
            "children": {}
        }
    }
    
    # Tree Menu 렌더링
    selected_menu = render_tree_menu(menu_structure)
    
    # API 키 상태 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 시스템 상태")
    
    if not OPENAI_API_KEY:
        st.sidebar.error("⚠️ OpenAI API 키가 설정되지 않았습니다.")
        st.sidebar.info("`.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
    else:
        st.sidebar.success("✅ OpenAI API 키가 설정되었습니다.")

    # 데이터 동기화 (임베딩 → D:/vector_store)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 데이터 동기화")
    st.sidebar.caption("프로젝트 임베딩을 벡터 스토어 에 저장합니다.")

    # 진행 상태용 플레이스홀더
    sync_status = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    sync_button = st.sidebar.button("임베딩 동기화 실행", use_container_width=True, type="primary")

    if sync_button:
        try:
            sync_status.info("임베딩 동기화 실행 중...")
            analyzer = SimilarityAnalyzer()
            total = analyzer.sync_vector_db()
            # 진행바는 단일 호출이라 완료로 표시
            progress_bar.progress(100)
            sync_status.success(f"✅ 동기화 완료: {total}건 임베딩 저장")
        except Exception as e:
            sync_status.error(f"동기화 실패: {e}")
            progress_bar.progress(0)
    
    return selected_menu

def render_tree_menu(menu_structure, level=0):
    """계층적 메뉴를 렌더링하는 재귀 함수"""
    # session_state 초기화
    if 'selected_menu' not in st.session_state:
        st.session_state.selected_menu = "recommendation"
    
    # 현재 레벨의 메뉴 항목들을 분류
    leaf_menus = {}  # 자식이 없는 메뉴들
    parent_menus = {}  # 자식이 있는 메뉴들
    
    for menu_name, menu_info in menu_structure.items():
        has_children = menu_info.get("children", {})
        if has_children:
            parent_menus[menu_name] = menu_info
        else:
            leaf_menus[menu_name] = menu_info
    
    # 자식이 없는 메뉴들을 버튼으로 표시
    if leaf_menus:
        for menu_name, menu_info in leaf_menus.items():
            menu_value = menu_info.get("value")
            is_selected = st.session_state.selected_menu == menu_value
            
            # 선택된 메뉴는 색깔이 바뀌는 버튼으로 표시
            if is_selected:
                if st.sidebar.button(menu_name, key=f"menu_{menu_name}_{level}", use_container_width=True, type="primary"):
                    st.session_state.selected_menu = menu_value
                    # URL 업데이트
                    st.query_params.page = menu_value
                    st.rerun()
            else:
                if st.sidebar.button(menu_name, key=f"menu_{menu_name}_{level}", use_container_width=True):
                    st.session_state.selected_menu = menu_value
                    # URL 업데이트
                    st.query_params.page = menu_value
                    st.rerun()
    
    # 자식이 있는 메뉴들을 expander로 표시
    for menu_name, menu_info in parent_menus.items():
        with st.sidebar.expander(menu_name, expanded=(level == 0)):
            child_selected = render_tree_menu(menu_info["children"], level + 1)
            if child_selected:
                return child_selected
    
    return st.session_state.selected_menu

# --- UI 컴포넌트 함수 ---
def display_recommendation_card(rec: Dict[str, Any], rank: int):
    """단일 추천 인원에 대한 정보를 카드 형태로 표시합니다."""
    st.markdown(f"""
    <div class="recommendation-card">
        <h3>🏆 {rank}. {rec['participant']}</h3>
        <div style="margin-bottom: 1rem;">
            <span class="score-badge" style="background-color: #28a745;">적합도: {rec['total_score']:.2f}</span>
            <span class="score-badge" style="background-color: #ffc107; color: #333;">최근 경험: {rec['recent_score']:.2f}</span>
            <span class="score-badge" style="background-color: #6c757d;">프로젝트 수: {rec['project_count']}개</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 추천 이유
    st.markdown("**💡 상세 추천 이유:**")
    for reason in rec['reasons']:
        st.markdown(f'<div class="reason-item">{reason}</div>', unsafe_allow_html=True)
    
    # 관련 프로젝트
    if rec['best_matches']:
        st.markdown("**📋 관련 프로젝트:**")
        for match in rec['best_matches']:
            project = match['project']
            score = match['score']
            st.markdown(f"""
            <div class="project-item">
                <strong>{project['프로젝트명']}</strong> (매칭도: {score:.2f})<br>
                기간: {project['프로젝트기간']}<br>
                태그: {', '.join(project['프로젝트태그'])}<br>
                개요: {project['프로젝트개요'][:100]}...
            </div>
            """, unsafe_allow_html=True)

# --- 각 페이지 렌더링 함수들 ---
def render_recommendation_page():
    """인원 추천 페이지를 렌더링합니다."""
    st.markdown("### 📝 프로젝트 설명 입력")
    user_input = st.text_area(
        "새로운 프로젝트에 대한 상세한 설명을 입력해주세요:",
        placeholder="예: AI 기반의 스마트 팩토리 구축 프로젝트로, 생산 라인 최적화 및 불량 예측 시스템 개발이 목표입니다.",
        height=150
    )

    if st.button("🚀 인원 추천 받기", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("프로젝트 설명을 입력해주세요.")
        else:
            with st.spinner("인원을 분석하고 추천 중입니다..."):
                try:
                    agent = load_agent()
                    if agent:
                        result = agent.process_request(user_input)
                        recommendations = result["recommendations"]
                        
                        # 디버깅 정보 출력
                        st.markdown("### 🔍 디버깅 정보")
                        st.write(f"**전체 결과 키들:** {list(result.keys())}")
                        st.write(f"**추천 수:** {len(recommendations)}")
                        st.write(f"**분석 완료:** {result.get('analysis_complete', 'N/A')}")
                        
                        if recommendations:
                            st.write("**추천 결과 상세:**")
                            for i, rec in enumerate(recommendations[:3]):
                                st.write(f"  {i+1}. {rec.get('participant', 'Unknown')} (점수: {rec.get('total_score', 0):.4f})")
                        
                        st.markdown("### 🎯 추천 결과")
                        
                        if recommendations:
                            for i, rec in enumerate(recommendations, 1):
                                display_recommendation_card(rec, i)
                        else:
                            st.info("요청하신 프로젝트에 적합한 인원을 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")

def render_statistics_page():
    """프로젝트 통계 페이지를 렌더링합니다."""
    st.markdown("### 📊 프로젝트 통계")
    st.markdown("전체 프로젝트 데이터에 대한 기본 통계 정보를 제공합니다.")
    
    data_loader = load_data()
    projects = data_loader.get_all_projects()
    
    if not projects:
        st.warning("로드된 프로젝트 데이터가 없습니다.")
        return
    
    # 기본 통계 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 프로젝트 수", len(projects))
    
    with col2:
        total_participants = len(data_loader.get_all_participants())
        st.metric("총 참여자 수", total_participants)
    
    with col3:
        recent_projects = 0
        for project in projects:
            if project.get('end_date'):
                from datetime import datetime, timedelta
                recent_threshold = datetime.now() - timedelta(days=180)
                if project['end_date'] >= recent_threshold:
                    recent_projects += 1
        st.metric("최근 6개월 프로젝트", recent_projects)
    
    with col4:
        total_participations = sum(len(project.get('참여자명단', [])) for project in projects)
        avg_participants = total_participations / len(projects) if projects else 0
        st.metric("프로젝트당 평균 참여자", f"{avg_participants:.1f}명")
    
    # 프로젝트 태그 통계
    st.markdown("#### 🏷️ 프로젝트 태그 통계")
    
    # 모든 태그 수집
    all_tags = []
    for project in projects:
        all_tags.extend(project.get('프로젝트태그', []))
    
    if all_tags:
        tag_counts = pd.Series(all_tags).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**태그 사용 빈도표**")
            tag_df = pd.DataFrame({
                '태그': tag_counts.head(10).index,
                '사용 횟수': tag_counts.head(10).values
            })
            st.dataframe(tag_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**태그 분포 원형 차트**")
            fig_pie = px.pie(
                values=tag_counts.head(10).values,
                names=tag_counts.head(10).index,
                title="상위 10개 태그 분포"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # 프로젝트 목록 (더블클릭 팝업 포함)
    st.markdown("#### 📋 프로젝트 목록")
    st.markdown("행을 더블클릭하면 프로젝트 상세 정보 팝업이 나타납니다.")
    
    df = pd.DataFrame(projects)
    display_columns = ['프로젝트명', '프로젝트기간', '프로젝트태그', '참여자명단']
    available_columns = [col for col in display_columns if col in df.columns]
    
    if available_columns:
        display_df = df[available_columns].copy()
        if '프로젝트태그' in display_df.columns:
            display_df['프로젝트태그'] = display_df['프로젝트태그'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        if '참여자명단' in display_df.columns:
            display_df['참여자명단'] = display_df['참여자명단'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        # 데이터프레임 표시 (체크박스 없이)
        selected_rows = st.dataframe(
            display_df, 
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True
        )
        
        # 선택된 행이 있으면 상세 정보 팝업 표시
        if selected_rows.selection.rows:
            selected_idx = selected_rows.selection.rows[0]
            selected_project = projects[selected_idx]
            
            # 팝업 스타일로 상세 정보 표시
            with st.container():
                st.markdown("---")
                st.markdown("### 📄 프로젝트 상세 정보")
                
                # 팝업 닫기 버튼
                if st.button("❌ 팝업 닫기", key="close_popup"):
                    st.rerun()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**프로젝트명:** {selected_project.get('프로젝트명', 'N/A')}")
                    st.markdown(f"**프로젝트 기간:** {selected_project.get('프로젝트기간', 'N/A')}")
                    st.markdown(f"**프로젝트 태그:** {', '.join(selected_project.get('프로젝트태그', []))}")
                    st.markdown(f"**참여자 명단:** {', '.join(selected_project.get('참여자명단', []))}")
                
                with col2:
                    st.markdown("**프로젝트 개요:**")
                    st.text_area(
                        "개요 내용",
                        value=selected_project.get('프로젝트개요', '개요 정보가 없습니다.'),
                        height=200,
                        disabled=True,
                        key=f"overview_{selected_idx}"
                    )

def render_analysis_page():
    """데이터 분석 페이지를 렌더링합니다."""
    st.markdown("### 📈 데이터 분석")
    st.markdown("프로젝트 데이터에 대한 종합적인 시각적 분석을 제공합니다.")
    
    data_loader = load_data()
    projects = data_loader.get_all_projects()
    
    if not projects:
        st.warning("로드된 프로젝트 데이터가 없습니다.")
        return

    # 1. 기본 통계 대시보드
    st.markdown("#### 📊 프로젝트 데이터 개요")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("총 프로젝트", len(projects))
    with col2:
        total_participants = len(data_loader.get_all_participants())
        st.metric("총 참여자", total_participants)
    with col3:
        total_tags = len(set([tag for project in projects for tag in project.get('프로젝트태그', [])]))
        st.metric("고유 태그 수", total_tags)
    with col4:
        avg_participants = sum(len(project.get('참여자명단', [])) for project in projects) / len(projects)
        st.metric("프로젝트당 평균 참여자", f"{avg_participants:.1f}명")
    with col5:
        recent_projects = len([p for p in projects if p.get('end_date') and 
                              (datetime.now() - p['end_date']).days <= 180])
        st.metric("최근 6개월 프로젝트", recent_projects)

    # 2. 프로젝트 기간 분석
    st.markdown("#### 📅 프로젝트 기간 분석")
    
    # 프로젝트 기간 데이터 처리
    project_durations = []
    project_years = []
    
    for project in projects:
        if project.get('start_date') and project.get('end_date'):
            duration_days = (project['end_date'] - project['start_date']).days
            project_durations.append(duration_days)
            project_years.append(project['start_date'].year)
    
    if project_durations:
        col1, col2 = st.columns(2)
        
        with col1:
            # 프로젝트 기간 분포
            duration_df = pd.DataFrame({'기간(일)': project_durations})
            fig_duration = px.histogram(
                duration_df, 
                x='기간(일)', 
                nbins=20,
                title='프로젝트 기간 분포',
                labels={'기간(일)': '프로젝트 기간 (일)', 'count': '프로젝트 수'}
            )
            st.plotly_chart(fig_duration, use_container_width=True)
        
        with col2:
            # 연도별 프로젝트 수
            year_counts = pd.Series(project_years).value_counts().sort_index()
            fig_years = px.bar(
                x=year_counts.index, 
                y=year_counts.values,
                title='연도별 프로젝트 수',
                labels={'x': '연도', 'y': '프로젝트 수'}
            )
            st.plotly_chart(fig_years, use_container_width=True)

    # 3. 태그 분석 (고급)
    st.markdown("#### 🏷️ 프로젝트 태그 심층 분석")
    
    all_tags = [tag for project in projects for tag in project.get('프로젝트태그', [])]
    if all_tags:
        tag_counts = pd.Series(all_tags).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 태그 사용 빈도 (상위 15개)
            tag_df = pd.DataFrame({
                '태그': tag_counts.head(15).index,
                '사용 횟수': tag_counts.head(15).values
            })
            fig_tags = px.bar(
                tag_df,
                x='사용 횟수',
                y='태그',
                orientation='h',
                title='태그 사용 빈도 (상위 15개)',
                labels={'사용 횟수': '사용 횟수', '태그': '태그'}
            )
            fig_tags.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_tags, use_container_width=True)
        
        with col2:
            # 태그 분포 원형 차트
            fig_pie = px.pie(
                values=tag_counts.head(10).values,
                names=tag_counts.head(10).index,
                title="상위 10개 태그 분포"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # 4. 참여자 분석 (고급)
    st.markdown("#### 👥 참여자 심층 분석")
    
    # 참여자별 프로젝트 수
    participant_project_counts = {}
    participant_tags = {}
    
    for project in projects:
        for participant in project.get('참여자명단', []):
            participant_project_counts[participant] = participant_project_counts.get(participant, 0) + 1
            if participant not in participant_tags:
                participant_tags[participant] = []
            participant_tags[participant].extend(project.get('프로젝트태그', []))
    
    if participant_project_counts:
        participant_df = pd.DataFrame(list(participant_project_counts.items()), columns=['참여자', '프로젝트수'])
        participant_df = participant_df.sort_values('프로젝트수', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 참여자별 프로젝트 수 (상위 15명)
            fig_participants = px.bar(
                participant_df.head(15),
                x='프로젝트수',
                y='참여자',
                orientation='h',
                title='참여자별 프로젝트 참여 수 (상위 15명)',
                labels={'프로젝트수': '참여 프로젝트 수', '참여자': '참여자'}
            )
            fig_participants.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_participants, use_container_width=True)
        
        with col2:
            # 참여자별 태그 다양성 (중복 태그 완전 제거: 전체 태그리스트에서 고유 태그만 카운트)
            participant_tag_diversity = {}
            for participant, tags in participant_tags.items():
                # 전체 태그리스트에서 고유 태그만 추출
                unique_tags = list(set(tags))
                participant_tag_diversity[participant] = len(unique_tags)
            diversity_df = pd.DataFrame(list(participant_tag_diversity.items()), columns=['참여자', '태그다양성'])
            diversity_df = diversity_df.sort_values('태그다양성', ascending=False).head(15)
            
            fig_diversity = px.bar(
                diversity_df,
                x='태그다양성',
                y='참여자',
                orientation='h',
                title='참여자별 태그 다양성 (상위 15명)',
                labels={'태그다양성': '고유 태그 수', '참여자': '참여자'}
            )
            fig_diversity.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_diversity, use_container_width=True)

    # 5. 협업 쌍 분석
    st.markdown("#### 🤝 협업 쌍 분석")
    
    # 모든 참여자 쌍의 협업 횟수 계산
    collaboration_pairs = {}
    
    for project in projects:
        participants = project.get('참여자명단', [])
        for i in range(len(participants)):
            for j in range(i + 1, len(participants)):
                pair = tuple(sorted([participants[i], participants[j]]))
                collaboration_pairs[pair] = collaboration_pairs.get(pair, 0) + 1
    
    if collaboration_pairs:
        # 상위 협업 쌍 추출
        top_collaborations = sorted(collaboration_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if top_collaborations:
            collab_df = pd.DataFrame([
                {'참여자1': pair[0], '참여자2': pair[1], '협업횟수': count}
                for pair, count in top_collaborations
            ])
            collab_df['협업쌍'] = collab_df['참여자1'] + ' & ' + collab_df['참여자2']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 협업 쌍 테이블
                st.markdown("**상위 협업 쌍**")
                st.dataframe(collab_df[['협업쌍', '협업횟수']], use_container_width=True, hide_index=True)
            
            with col2:
                # 협업 쌍 차트
                fig_collab = px.bar(
                    collab_df.head(10),
                    x='협업횟수',
                    y='협업쌍',
                    orientation='h',
                    title='상위 10개 협업 쌍',
                    labels={'협업횟수': '협업 횟수', '협업쌍': '협업 쌍'}
                )
                fig_collab.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_collab, use_container_width=True)

    # 6. 워드 클라우드
    st.markdown("#### ☁️ 프로젝트 개요 워드 클라우드")
    
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 모든 프로젝트 개요 텍스트 수집
        all_overviews = ' '.join([project.get('프로젝트개요', '') for project in projects])
        
        if all_overviews.strip():
            # 한국어 폰트 찾기
            def find_korean_font():
                """시스템에서 사용 가능한 한국어 폰트를 찾습니다."""
                # Windows에서 일반적으로 사용되는 한국어 폰트들
                korean_fonts = [
                    'C:/Windows/Fonts/malgun.ttf',  # 맑은 고딕
                    'C:/Windows/Fonts/gulim.ttc',   # 굴림
                    'C:/Windows/Fonts/batang.ttc',  # 바탕
                    'C:/Windows/Fonts/arial.ttf',   # Arial (영어만)
                ]
                
                for font_path in korean_fonts:
                    if os.path.exists(font_path):
                        return font_path
                
                # matplotlib에서 사용 가능한 폰트 중 한국어 지원 폰트 찾기
                font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                for font_path in font_list:
                    try:
                        font_prop = fm.FontProperties(fname=font_path)
                        font_name = font_prop.get_name()
                        # 한국어 폰트 이름 패턴 확인
                        if any(korean in font_name.lower() for korean in ['malgun', 'gulim', 'batang', 'dotum', 'gungsuh']):
                            return font_path
                    except:
                        continue
                
                return None
            
            # 한국어 폰트 경로 찾기
            font_path = find_korean_font()
            
            if font_path:
                # 워드 클라우드 생성 (한국어 폰트 적용)
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100,
                    colormap='viridis',
                    font_path=font_path,
                    prefer_horizontal=0.9,  # 가로 배치 선호
                    relative_scaling=0.5,   # 크기 차이 조정
                    min_font_size=10,       # 최소 폰트 크기
                    max_font_size=200       # 최대 폰트 크기
                ).generate(all_overviews)
                
                # 워드 클라우드 표시
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('프로젝트 개요 워드 클라우드', fontsize=16, pad=20)
                
                # matplotlib 한글 폰트 설정
                plt.rcParams['font.family'] = 'DejaVu Sans'
                if font_path:
                    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
                
                st.pyplot(fig)
            else:
                st.warning("한국어 폰트를 찾을 수 없습니다. 기본 폰트로 표시됩니다.")
                
                # 기본 폰트로 워드 클라우드 생성
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100,
                    colormap='viridis',
                    prefer_horizontal=0.9,
                    relative_scaling=0.5,
                    min_font_size=10,
                    max_font_size=200
                ).generate(all_overviews)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        else:
            st.info("프로젝트 개요 데이터가 없어 워드 클라우드를 생성할 수 없습니다.")
            
    except ImportError:
        st.warning("워드 클라우드를 표시하려면 `wordcloud` 패키지가 필요합니다. 설치하려면: `pip install wordcloud`")
        
        # 대안: 단어 빈도 분석
        st.markdown("**대안: 단어 빈도 분석**")
        import re
        from collections import Counter
        
        # 한국어 텍스트 전처리
        korean_text = re.sub(r'[^가-힣\s]', '', all_overviews)
        words = [word for word in korean_text.split() if len(word) > 1]
        
        if words:
            word_counts = Counter(words)
            top_words = word_counts.most_common(20)
            
            word_df = pd.DataFrame(top_words, columns=['단어', '빈도'])
            fig_words = px.bar(
                word_df,
                x='빈도',
                y='단어',
                orientation='h',
                title='프로젝트 개요 상위 단어 (20개)',
                labels={'빈도': '빈도', '단어': '단어'}
            )
            fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)

    # 7. 프로젝트 복잡도 분석
    st.markdown("#### 📊 프로젝트 복잡도 분석")
    
    # 프로젝트별 복잡도 지표 계산
    complexity_data = []
    for project in projects:
        participant_count = len(project.get('참여자명단', []))
        tag_count = len(project.get('프로젝트태그', []))
        overview_length = len(project.get('프로젝트개요', ''))
        
        # 복잡도 점수 (참여자 수 + 태그 수 + 개요 길이/100)
        complexity_score = participant_count + tag_count + (overview_length / 100)
        
        complexity_data.append({
            '프로젝트명': project.get('프로젝트명', ''),
            '참여자수': participant_count,
            '태그수': tag_count,
            '개요길이': overview_length,
            '복잡도점수': complexity_score
        })
    
    if complexity_data:
        complexity_df = pd.DataFrame(complexity_data)
        complexity_df = complexity_df.sort_values('복잡도점수', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 복잡도 점수 분포
            fig_complexity = px.histogram(
                complexity_df,
                x='복잡도점수',
                nbins=15,
                title='프로젝트 복잡도 점수 분포',
                labels={'복잡도점수': '복잡도 점수', 'count': '프로젝트 수'}
            )
            st.plotly_chart(fig_complexity, use_container_width=True)
        
        with col2:
            # 상위 복잡한 프로젝트
            st.markdown("**가장 복잡한 프로젝트 (상위 10개)**")
            top_complex = complexity_df.head(10)[['프로젝트명', '참여자수', '태그수', '복잡도점수']]
            st.dataframe(top_complex, use_container_width=True, hide_index=True)

def render_help_page():
    """도움말 페이지를 렌더링합니다."""
    st.markdown("### ❓ 도움말")
    
    st.markdown("""
    #### 🤖 프로젝트 인원 추천 챗봇 사용법
    
    **1. 🚀 인원 추천**
    - 새로운 프로젝트에 대한 상세한 설명을 입력하세요
    - AI가 과거 프로젝트 데이터를 분석하여 적합한 인원을 추천합니다
    - 추천 이유와 관련 프로젝트 경험을 상세히 확인할 수 있습니다
    
    **2. 📊 프로젝트 통계**
    - 전체 프로젝트 데이터의 기본 통계를 확인할 수 있습니다
    - 총 프로젝트 수, 참여자 수, 최근 프로젝트 현황 등을 제공합니다
    - 프로젝트 목록을 테이블 형태로 확인할 수 있습니다
    
    **3. 📈 데이터 분석**
    - 프로젝트 데이터에 대한 시각적 분석을 제공합니다
    - 인기 태그, 참여자별 활동 현황 등을 차트로 확인할 수 있습니다
    
    #### 💡 사용 팁
    
    - **프로젝트 설명 작성 시**: 구체적인 기술, 도메인, 목표를 포함하면 더 정확한 추천을 받을 수 있습니다
    - **추천 결과 해석**: 적합도 점수는 0~1 사이의 값으로, 높을수록 더 적합한 인원입니다
    - **최근 경험**: 최근 6개월 내 관련 프로젝트 경험이 있는 인원에게 가중치가 부여됩니다
    
    #### 🔧 기술 스택
    
    - **AI 모델**: OpenAI GPT-4o-mini
    - **유사도 분석**: Sentence Transformers (다국어 지원)
    - **UI 프레임워크**: Streamlit
    - **워크플로우**: LangGraph
    
    #### 📞 문의사항
    
    추가 기능이나 개선사항이 있으시면 개발팀에 문의해주세요.
    """)

# --- URL 라우팅 함수 ---
def get_page_from_url():
    """URL 쿼리 파라미터에서 페이지 정보를 가져옵니다."""
    query_params = st.query_params
    page = query_params.get("page", "recommendation")
    
    # 유효한 페이지인지 확인
    valid_pages = ["recommendation", "statistics", "analysis", "help"]
    if page not in valid_pages:
        page = "recommendation"
    
    return page

def update_url(page):
    """URL을 업데이트합니다."""
    st.query_params.page = page

# --- 메인 애플리케이션 함수 ---
def main():
    """Streamlit 애플리케이션의 메인 함수입니다."""
    # URL에서 페이지 정보 가져오기
    url_page = get_page_from_url()
    
    # session_state 초기화
    if 'selected_menu' not in st.session_state:
        st.session_state.selected_menu = url_page
    
    # 헤더
    st.markdown("<h1 class='main-header'>🤖 프로젝트 인원 추천 챗봇</h1>", unsafe_allow_html=True)
    
    # 사이드바 메뉴
    selected_page = render_sidebar()
    
    # URL과 session_state 동기화
    if selected_page != url_page:
        update_url(selected_page)
    
    # 메인 콘텐츠 영역
    st.markdown("---")
    
    # 선택된 페이지에 따라 콘텐츠 렌더링
    if selected_page == "recommendation":
        render_recommendation_page()
    elif selected_page == "statistics":
        render_statistics_page()
    elif selected_page == "analysis":
        render_analysis_page()
    elif selected_page == "help":
        render_help_page()

if __name__ == "__main__":
    main()
