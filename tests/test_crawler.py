"""データ収集・パース・クレンジングモジュールの単体テスト"""

import pytest
from src.crawler.html_parser import RaceHtmlParser
from src.pipeline.cleaner import DataCleaner


def test_cleaner_extract_int():
    """数値抽出処理の単体テスト"""
    assert DataCleaner._extract_int("1着") == 1
    assert DataCleaner._extract_int("8") == 8
    assert DataCleaner._extract_int("取") is None


def test_cleaner_parse_time_to_seconds():
    """タイム（分:秒）から総秒数への変換テスト"""
    assert DataCleaner._parse_time_to_seconds("1:33.4") == 93.4
    assert DataCleaner._parse_time_to_seconds("58.2") == 58.2
    assert DataCleaner._parse_time_to_seconds(None) is None


def test_html_parser_empty_input():
    """空HTML入力時の例外発生テスト"""
    from src.common.exceptions import ParseError
    with pytest.raises(ParseError):
        RaceHtmlParser.parse_race_result("", "dummy_id")