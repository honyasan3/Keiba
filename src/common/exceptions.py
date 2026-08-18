"""システム共通のカスタム例外クラス定義"""

class KeibaAIError(Exception):
    """競馬予想AIシステムの基底例外クラス"""
    pass

class ConfigurationError(KeibaAIError):
    """設定ファイルのロード・バリデーションエラー"""
    pass

class ScraperError(KeibaAIError):
    """データ収集・スクレイピング処理中のエラー"""
    pass

class ParseError(KeibaAIError):
    """HTML・データのパース処理中のエラー"""
    pass

class DatabaseError(KeibaAIError):
    """データベース接続・操作エラー"""
    pass

class DataLeakageError(KeibaAIError):
    """時系列リーク検知時のエラー"""
    pass