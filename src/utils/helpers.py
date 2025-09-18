"""
도우미 함수 모듈
===============

이 모듈은 애플리케이션 전반에서 사용되는 일반적인 도우미 함수들을 제공합니다.

주요 기능:
- 데이터 변환 및 포맷팅
- 파일 처리
- 문자열 처리
- 날짜/시간 처리

작성자: AI Assistant
버전: 1.0.0
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

def ensure_directory_exists(directory_path: str) -> bool:
    """
    디렉토리가 존재하는지 확인하고, 없으면 생성하는 함수
    
    Args:
        directory_path (str): 확인할 디렉토리 경로
        
    Returns:
        bool: 디렉토리 생성 성공 여부
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return True
    except Exception as e:
        print(f"디렉토리 생성 실패: {e}")
        return False

def safe_json_load(file_path: str, default: Any = None) -> Any:
    """
    JSON 파일을 안전하게 로드하는 함수
    
    Args:
        file_path (str): JSON 파일 경로
        default (Any): 로드 실패 시 반환할 기본값
        
    Returns:
        Any: 로드된 JSON 데이터 또는 기본값
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON 파일 로드 실패: {e}")
        return default

def safe_json_save(data: Any, file_path: str) -> bool:
    """
    데이터를 JSON 파일로 안전하게 저장하는 함수
    
    Args:
        data (Any): 저장할 데이터
        file_path (str): 저장할 파일 경로
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"JSON 파일 저장 실패: {e}")
        return False

def format_score(score: float, decimal_places: int = 2) -> str:
    """
    점수를 포맷팅하는 함수
    
    Args:
        score (float): 포맷팅할 점수
        decimal_places (int): 소수점 자릿수
        
    Returns:
        str: 포맷팅된 점수 문자열
    """
    return f"{score:.{decimal_places}f}"

def format_percentage(value: float, total: float, decimal_places: int = 1) -> str:
    """
    백분율을 포맷팅하는 함수
    
    Args:
        value (float): 값
        total (float): 전체값
        decimal_places (int): 소수점 자릿수
        
    Returns:
        str: 포맷팅된 백분율 문자열
    """
    if total == 0:
        return "0.0%"
    percentage = (value / total) * 100
    return f"{percentage:.{decimal_places}f}%"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    텍스트를 지정된 길이로 자르는 함수
    
    Args:
        text (str): 자를 텍스트
        max_length (int): 최대 길이
        suffix (str): 자른 후 추가할 접미사
        
    Returns:
        str: 잘린 텍스트
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def parse_date_string(date_str: str, format_str: str = "%Y.%m") -> Optional[datetime]:
    """
    날짜 문자열을 파싱하는 함수
    
    Args:
        date_str (str): 파싱할 날짜 문자열
        format_str (str): 날짜 형식
        
    Returns:
        Optional[datetime]: 파싱된 날짜 객체 또는 None
    """
    try:
        return datetime.strptime(date_str, format_str)
    except ValueError:
        return None

def calculate_months_difference(start_date: datetime, end_date: datetime) -> int:
    """
    두 날짜 간의 개월 차이를 계산하는 함수
    
    Args:
        start_date (datetime): 시작 날짜
        end_date (datetime): 종료 날짜
        
    Returns:
        int: 개월 차이
    """
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

def clean_text(text: str) -> str:
    """
    텍스트를 정리하는 함수 (공백 제거, 특수문자 정리 등)
    
    Args:
        text (str): 정리할 텍스트
        
    Returns:
        str: 정리된 텍스트
    """
    if not text:
        return ""
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    # 연속된 공백을 하나로 변경
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text

def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """
    텍스트에서 키워드를 추출하는 함수
    
    Args:
        text (str): 키워드를 추출할 텍스트
        min_length (int): 최소 키워드 길이
        
    Returns:
        List[str]: 추출된 키워드 리스트
    """
    if not text:
        return []
    
    import re
    
    # 한글, 영문, 숫자만 추출
    words = re.findall(r'[가-힣a-zA-Z0-9]+', text)
    
    # 최소 길이 이상인 단어만 필터링
    keywords = [word for word in words if len(word) >= min_length]
    
    return keywords

def create_summary_stats(data: List[Dict[str, Any]], key_field: str) -> Dict[str, Any]:
    """
    데이터 리스트에서 통계 요약을 생성하는 함수
    
    Args:
        data (List[Dict[str, Any]]): 통계를 생성할 데이터 리스트
        key_field (str): 통계를 생성할 필드명
        
    Returns:
        Dict[str, Any]: 통계 요약 정보
    """
    if not data:
        return {"count": 0, "unique": 0, "most_common": []}
    
    values = [item.get(key_field, "") for item in data if key_field in item]
    
    from collections import Counter
    counter = Counter(values)
    
    return {
        "count": len(values),
        "unique": len(counter),
        "most_common": counter.most_common(10)
    }

def validate_email(email: str) -> bool:
    """
    이메일 주소 형식을 검증하는 함수
    
    Args:
        email (str): 검증할 이메일 주소
        
    Returns:
        bool: 유효한 이메일 주소 여부
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """
    전화번호 형식을 검증하는 함수 (한국 형식)
    
    Args:
        phone (str): 검증할 전화번호
        
    Returns:
        bool: 유효한 전화번호 여부
    """
    import re
    # 한국 전화번호 패턴 (010-1234-5678, 02-123-4567 등)
    pattern = r'^(\d{2,3})-(\d{3,4})-(\d{4})$'
    return bool(re.match(pattern, phone))
