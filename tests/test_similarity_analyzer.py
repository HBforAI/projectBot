"""
유사도 분석기 테스트 모듈
========================

이 모듈은 SimilarityAnalyzer 클래스의 기능을 테스트합니다.

테스트 항목:
- 텍스트 유사도 계산
- 태그 유사도 계산
- 프로젝트 유사도 계산
- 시간 가중치 계산
- 참여자 적합도 계산

작성자: AI Assistant
버전: 1.0.0
"""

import unittest
import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.similarity_analyzer import SimilarityAnalyzer

class TestSimilarityAnalyzer(unittest.TestCase):
    """
    SimilarityAnalyzer 클래스의 테스트 케이스
    """
    
    def setUp(self):
        """
        테스트 설정 메서드
        각 테스트 메서드 실행 전에 호출됩니다.
        """
        self.analyzer = SimilarityAnalyzer()
    
    def test_calculate_similarity(self):
        """
        텍스트 유사도 계산 테스트
        """
        # 동일한 텍스트
        similarity = self.analyzer.calculate_similarity("AI 프로젝트", "AI 프로젝트")
        self.assertEqual(similarity, 1.0)
        
        # 유사한 텍스트
        similarity = self.analyzer.calculate_similarity("AI 기반 고객 서비스", "AI 고객 서비스 개선")
        self.assertGreater(similarity, 0.5)
        
        # 다른 텍스트
        similarity = self.analyzer.calculate_similarity("AI 프로젝트", "HR 관리 시스템")
        self.assertLess(similarity, 0.5)
        
        # 빈 텍스트
        similarity = self.analyzer.calculate_similarity("", "AI 프로젝트")
        self.assertEqual(similarity, 0.0)
    
    def test_calculate_tag_similarity(self):
        """
        태그 유사도 계산 테스트
        """
        # 동일한 태그
        similarity = self.analyzer.calculate_tag_similarity(["AI", "고객서비스"], ["AI", "고객서비스"])
        self.assertGreater(similarity, 0.8)
        
        # 유사한 태그
        similarity = self.analyzer.calculate_tag_similarity(["AI", "고객"], ["AI", "고객서비스"])
        self.assertGreater(similarity, 0.5)
        
        # 다른 태그
        similarity = self.analyzer.calculate_tag_similarity(["AI"], ["HR", "관리"])
        self.assertLess(similarity, 0.5)
        
        # 빈 태그
        similarity = self.analyzer.calculate_tag_similarity([], ["AI"])
        self.assertEqual(similarity, 0.0)
    
    def test_extract_keywords_from_text(self):
        """
        키워드 추출 테스트
        """
        # 일반적인 텍스트
        keywords = self.analyzer.extract_keywords_from_text("AI 기반 고객 서비스 개선 프로젝트")
        self.assertIn("AI", keywords)
        self.assertIn("고객", keywords)
        self.assertIn("서비스", keywords)
        
        # 빈 텍스트
        keywords = self.analyzer.extract_keywords_from_text("")
        self.assertEqual(len(keywords), 0)
        
        # 특수문자가 포함된 텍스트
        keywords = self.analyzer.extract_keywords_from_text("AI@기반#고객$서비스%개선")
        self.assertGreater(len(keywords), 0)
    
    def test_calculate_project_similarity(self):
        """
        프로젝트 유사도 계산 테스트
        """
        project = {
            '프로젝트명': 'AI 기반 고객 서비스 개선',
            '프로젝트개요': '고객 서비스를 AI 기술로 개선하는 프로젝트',
            '프로젝트태그': ['AI', '고객서비스', '개선']
        }
        
        user_request = "AI 고객 서비스 프로젝트"
        
        similarities = self.analyzer.calculate_project_similarity(user_request, project)
        
        # 반환값이 딕셔너리인지 확인
        self.assertIsInstance(similarities, dict)
        
        # 필요한 키들이 있는지 확인
        required_keys = ['title', 'overview', 'tags']
        for key in required_keys:
            self.assertIn(key, similarities)
        
        # 유사도 값이 0과 1 사이인지 확인
        for value in similarities.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_calculate_time_weight(self):
        """
        시간 가중치 계산 테스트
        """
        # 최근 프로젝트 (6개월 이내)
        recent_project = {
            '프로젝트기간': '2024.01 ~ 2024.06'
        }
        weight = self.analyzer.calculate_time_weight(recent_project)
        self.assertEqual(weight, 1.5)
        
        # 1년 이내 프로젝트
        year_project = {
            '프로젝트기간': '2023.01 ~ 2023.12'
        }
        weight = self.analyzer.calculate_time_weight(year_project)
        self.assertEqual(weight, 1.2)
        
        # 2년 이상 프로젝트
        old_project = {
            '프로젝트기간': '2020.01 ~ 2020.12'
        }
        weight = self.analyzer.calculate_time_weight(old_project)
        self.assertEqual(weight, 0.8)
        
        # 잘못된 형식의 날짜
        invalid_project = {
            '프로젝트기간': 'invalid-date'
        }
        weight = self.analyzer.calculate_time_weight(invalid_project)
        self.assertEqual(weight, 1.0)
    
    def test_calculate_participant_suitability(self):
        """
        참여자 적합도 계산 테스트
        """
        participant_name = "테스트 참여자"
        participant_projects = [
            {
                '프로젝트명': 'AI 기반 고객 서비스 개선',
                '프로젝트개요': '고객 서비스를 AI 기술로 개선하는 프로젝트',
                '프로젝트태그': ['AI', '고객서비스', '개선'],
                '프로젝트기간': '2024.01 ~ 2024.06'
            }
        ]
        
        user_request = "AI 고객 서비스 프로젝트"
        
        suitability = self.analyzer.calculate_participant_suitability(
            user_request, participant_name, participant_projects
        )
        
        # 반환값이 딕셔너리인지 확인
        self.assertIsInstance(suitability, dict)
        
        # 필요한 키들이 있는지 확인
        required_keys = ['participant', 'total_score', 'recent_score', 'project_count', 'recent_project_count', 'best_matches', 'reasons']
        for key in required_keys:
            self.assertIn(key, suitability)
        
        # 참여자 이름이 올바른지 확인
        self.assertEqual(suitability['participant'], participant_name)
        
        # 점수가 0과 1 사이인지 확인
        self.assertGreaterEqual(suitability['total_score'], 0.0)
        self.assertLessEqual(suitability['total_score'], 1.0)
        
        # 프로젝트 수가 올바른지 확인
        self.assertEqual(suitability['project_count'], len(participant_projects))
        
        # 추천 이유가 리스트인지 확인
        self.assertIsInstance(suitability['reasons'], list)
    
    def test_empty_participant_projects(self):
        """
        빈 참여자 프로젝트 리스트 테스트
        """
        participant_name = "테스트 참여자"
        participant_projects = []
        
        user_request = "AI 고객 서비스 프로젝트"
        
        suitability = self.analyzer.calculate_participant_suitability(
            user_request, participant_name, participant_projects
        )
        
        # 빈 프로젝트 리스트에 대한 처리 확인
        self.assertEqual(suitability['total_score'], 0.0)
        self.assertEqual(suitability['project_count'], 0)
        self.assertEqual(len(suitability['reasons']), 1)
        self.assertIn("관련 프로젝트 경험이 없습니다", suitability['reasons'][0])

if __name__ == '__main__':
    # 테스트 실행
    unittest.main()
