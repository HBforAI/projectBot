"""
데이터 로더 테스트 모듈
=====================

이 모듈은 ProjectDataLoader 클래스의 기능을 테스트합니다.

테스트 항목:
- 데이터 로드 기능
- 참여자별 프로젝트 조회
- 최근 프로젝트 조회
- 통계 정보 생성

작성자: AI Assistant
버전: 1.0.0
"""

import unittest
import sys
import os
from datetime import datetime

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.data_loader import ProjectDataLoader

class TestProjectDataLoader(unittest.TestCase):
    """
    ProjectDataLoader 클래스의 테스트 케이스
    """
    
    def setUp(self):
        """
        테스트 설정 메서드
        각 테스트 메서드 실행 전에 호출됩니다.
        """
        self.data_loader = ProjectDataLoader()
    
    def test_data_loading(self):
        """
        데이터 로드 기능 테스트
        """
        projects = self.data_loader.get_all_projects()
        self.assertIsInstance(projects, list)
        self.assertGreater(len(projects), 0, "프로젝트 데이터가 로드되지 않았습니다.")
    
    def test_participant_projects(self):
        """
        참여자별 프로젝트 조회 테스트
        """
        participants = self.data_loader.get_all_participants()
        self.assertIsInstance(participants, list)
        self.assertGreater(len(participants), 0, "참여자 데이터가 없습니다.")
        
        # 첫 번째 참여자의 프로젝트 조회 테스트
        if participants:
            participant_projects = self.data_loader.get_projects_by_participant(participants[0])
            self.assertIsInstance(participant_projects, list)
    
    def test_recent_projects(self):
        """
        최근 프로젝트 조회 테스트
        """
        recent_projects = self.data_loader.get_recent_projects(12)
        self.assertIsInstance(recent_projects, list)
        
        # 최근 프로젝트는 전체 프로젝트 수보다 작거나 같아야 함
        all_projects = self.data_loader.get_all_projects()
        self.assertLessEqual(len(recent_projects), len(all_projects))
    
    def test_project_tags(self):
        """
        프로젝트 태그 조회 테스트
        """
        tags = self.data_loader.get_project_tags()
        self.assertIsInstance(tags, list)
        self.assertGreater(len(tags), 0, "태그 데이터가 없습니다.")
    
    def test_participant_statistics(self):
        """
        참여자 통계 생성 테스트
        """
        stats = self.data_loader.get_participant_statistics()
        self.assertIsInstance(stats, list)
        self.assertGreater(len(stats), 0, "참여자 통계가 생성되지 않았습니다.")
        
        # 통계 데이터 구조 검증
        if stats:
            stat = stats[0]
            required_fields = ['참여자', '총_프로젝트', '최근_프로젝트', '활동도']
            for field in required_fields:
                self.assertIn(field, stat, f"통계 데이터에 {field} 필드가 없습니다.")
    
    def test_project_statistics(self):
        """
        프로젝트 통계 생성 테스트
        """
        stats = self.data_loader.get_project_statistics()
        self.assertIsInstance(stats, dict)
        
        required_fields = ['총_프로젝트_수', '총_참여자_수', '총_태그_수']
        for field in required_fields:
            self.assertIn(field, stats, f"통계 데이터에 {field} 필드가 없습니다.")
    
    def test_data_structure(self):
        """
        데이터 구조 검증 테스트
        """
        projects = self.data_loader.get_all_projects()
        
        if projects:
            project = projects[0]
            required_fields = ['프로젝트명', '프로젝트기간', '프로젝트개요', '프로젝트태그', '참여자명단']
            
            for field in required_fields:
                self.assertIn(field, project, f"프로젝트 데이터에 {field} 필드가 없습니다.")
            
            # 참여자명단은 리스트여야 함
            self.assertIsInstance(project['참여자명단'], list)
            
            # 프로젝트태그는 리스트여야 함
            self.assertIsInstance(project['프로젝트태그'], list)

if __name__ == '__main__':
    # 테스트 실행
    unittest.main()
