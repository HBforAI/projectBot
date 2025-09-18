"""
프로젝트 데이터 로더 모듈
========================

이 모듈은 JSON 형태의 프로젝트 데이터를 로드하고 전처리하는 기능을 제공합니다.
프로젝트 정보, 참여자 정보, 태그 정보 등을 효율적으로 관리합니다.

주요 기능:
- JSON 데이터 로드 및 파싱
- 참여자별 프로젝트 필터링
- 최근 프로젝트 조회
- 통계 정보 제공

작성자: AI Assistant
버전: 1.0.0
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from .config import PROJECT_DATA_PATH

class ProjectDataLoader:
    """
    프로젝트 데이터를 로드하고 관리하는 클래스
    
    이 클래스는 JSON 파일에서 프로젝트 데이터를 읽어와서
    다양한 방식으로 조회할 수 있는 메서드들을 제공합니다.
    """
    
    def __init__(self):
        """
        데이터 로더 초기화
        
        생성자에서 자동으로 프로젝트 데이터를 로드합니다.
        """
        self.projects = []  # 프로젝트 데이터를 저장할 리스트
        self.load_data()    # 데이터 로드 실행
    
    def load_data(self):
        """
        JSON 파일에서 프로젝트 데이터를 로드하는 메서드
        
        프로젝트 데이터 구조:
        - 프로젝트명: 프로젝트 제목
        - 프로젝트기간: 시작일 ~ 종료일 (YYYY.MM 형식)
        - 프로젝트개요: 프로젝트 상세 설명
        - 프로젝트태그: 관련 키워드 배열
        - 참여자명단: 프로젝트 참여자들 이름 배열
        """
        try:
            with open(PROJECT_DATA_PATH, 'r', encoding='utf-8') as f:
                self.projects = json.load(f)
            print(f"✅ 총 {len(self.projects)}개의 프로젝트 데이터를 로드했습니다.")
        except FileNotFoundError:
            print(f"❌ 프로젝트 데이터 파일을 찾을 수 없습니다: {PROJECT_DATA_PATH}")
            self.projects = []
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파일 형식이 올바르지 않습니다: {e}")
            self.projects = []
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {e}")
            self.projects = []
    
    def get_all_projects(self) -> List[Dict[str, Any]]:
        """
        모든 프로젝트 데이터를 반환하는 메서드
        
        Returns:
            List[Dict[str, Any]]: 모든 프로젝트 정보 리스트
        """
        return self.projects
    
    def get_projects_by_participant(self, participant_name: str) -> List[Dict[str, Any]]:
        """
        특정 참여자가 참여한 프로젝트들을 반환하는 메서드
        
        Args:
            participant_name (str): 참여자 이름
            
        Returns:
            List[Dict[str, Any]]: 해당 참여자가 참여한 프로젝트 리스트
        """
        return [project for project in self.projects 
                if participant_name in project.get('참여자명단', [])]
    
    def get_recent_projects(self, months: int = 12) -> List[Dict[str, Any]]:
        """
        최근 N개월 내에 완료된 프로젝트들을 반환하는 메서드
        
        Args:
            months (int): 조회할 개월 수 (기본값: 12개월)
            
        Returns:
            List[Dict[str, Any]]: 최근 프로젝트 리스트
        """
        current_date = datetime.now()
        recent_projects = []
        
        for project in self.projects:
            try:
                # 프로젝트 종료일 파싱 (YYYY.MM 형식)
                end_date_str = project.get('프로젝트기간', '').split(' ~ ')[-1]
                if end_date_str:
                    end_date = datetime.strptime(end_date_str, '%Y.%m')
                    months_diff = (current_date.year - end_date.year) * 12 + (current_date.month - end_date.month)
                    
                    if months_diff <= months:
                        recent_projects.append(project)
            except (ValueError, IndexError):
                # 날짜 파싱 실패 시 해당 프로젝트는 무시
                continue
        
        return recent_projects
    
    def get_all_participants(self) -> List[str]:
        """
        모든 참여자 목록을 반환하는 메서드 (중복 제거)
        
        Returns:
            List[str]: 참여자 이름 리스트 (알파벳 순으로 정렬)
        """
        participants = set()
        for project in self.projects:
            participants.update(project.get('참여자명단', []))
        return sorted(list(participants))
    
    def get_project_tags(self) -> List[str]:
        """
        모든 프로젝트 태그 목록을 반환하는 메서드 (중복 제거)
        
        Returns:
            List[str]: 태그 리스트 (알파벳 순으로 정렬)
        """
        tags = set()
        for project in self.projects:
            tags.update(project.get('프로젝트태그', []))
        return sorted(list(tags))
    
    def get_participant_statistics(self) -> Dict[str, Any]:
        """
        참여자별 통계 정보를 반환하는 메서드
        
        Returns:
            Dict[str, Any]: 참여자별 통계 정보
        """
        participants = self.get_all_participants()
        stats = []
        
        for participant in participants:
            projects = self.get_projects_by_participant(participant)
            recent_projects = self.get_recent_projects(12)
            recent_count = len([p for p in projects if p in recent_projects])
            
            stats.append({
                '참여자': participant,
                '총_프로젝트': len(projects),
                '최근_프로젝트': recent_count,
                '활동도': len(projects) / len(self.projects) * 100 if self.projects else 0
            })
        
        return sorted(stats, key=lambda x: x['총_프로젝트'], reverse=True)
    
    def get_project_statistics(self) -> Dict[str, Any]:
        """
        프로젝트 전체 통계 정보를 반환하는 메서드
        
        Returns:
            Dict[str, Any]: 프로젝트 통계 정보
        """
        all_tags = []
        years = []
        
        for project in self.projects:
            # 태그 수집
            all_tags.extend(project.get('프로젝트태그', []))
            
            # 연도 수집
            try:
                year = project.get('프로젝트기간', '').split(' ~ ')[0].split('.')[0]
                if year.isdigit():
                    years.append(int(year))
            except:
                continue
        
        return {
            '총_프로젝트_수': len(self.projects),
            '총_참여자_수': len(self.get_all_participants()),
            '총_태그_수': len(set(all_tags)),
            '태그_분포': pd.Series(all_tags).value_counts().to_dict(),
            '연도_분포': pd.Series(years).value_counts().to_dict()
        }
