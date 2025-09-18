#!/usr/bin/env python3
"""
테스트 실행 스크립트
==================

이 스크립트는 프로젝트의 모든 테스트를 실행합니다.

사용법:
    python run_tests.py

작성자: AI Assistant
버전: 1.0.0
"""

import unittest
import sys
import os
from pathlib import Path

def run_all_tests():
    """
    모든 테스트를 실행하는 함수
    
    Returns:
        bool: 모든 테스트가 통과하면 True
    """
    # 현재 디렉토리를 스크립트가 있는 디렉토리로 변경
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 테스트 디렉토리를 Python 경로에 추가
    sys.path.insert(0, str(script_dir))
    
    # 테스트 로더 생성
    loader = unittest.TestLoader()
    
    # tests 디렉토리에서 모든 테스트 발견
    test_suite = loader.discover('tests', pattern='test_*.py')
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def main():
    """
    메인 함수
    """
    print("🧪 프로젝트 테스트를 시작합니다...\n")
    
    success = run_all_tests()
    
    if success:
        print("\n✅ 모든 테스트가 성공했습니다!")
    else:
        print("\n❌ 일부 테스트가 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()
