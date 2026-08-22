"""当日の全レースを対象に、各レースの発走時刻の少し前にpredict.pyを自動実行する
Windowsタスクスケジューラのタスクを登録するセットアップスクリプト。

このスクリプト自体は「登録」だけを行う。実際のpredict.py実行は各レースの発走時刻近くに
タスクスケジューラが独立してトリガーするため、このプロセスが長時間起動し続ける必要はなく、
このセッション（ターミナル）を閉じても登録済みタスクは影響を受けない。

発走時刻の取得には専用の軽量フェッチを使う（BaseScraperのfetch_page/fetch_race_cardは
呼び出すと結果をキャッシュしてしまい、発走直前の実予想実行時に何時間も前の古い出馬表
スナップショットを読んでしまう＝直前の馬体重確定・出走除外・騎手変更などを取りこぼす
ため、意図的に避けている）。
"""
import argparse
import datetime
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.crawler.race_schedule_scraper import RaceScheduleScraper
from src.common.logger import setup_logger

logger = setup_logger("schedule_today_predictions")

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
LOG_DIR = PROJECT_DIR / "logs" / "scheduled"
TASK_PREFIX = "KeibaPredict_"
WRAPPER_BAT = PROJECT_DIR / "run_scheduled_predict.bat"


def ensure_wrapper_bat() -> None:
    """schtasksの/TRは261文字までしか受け付けないため、日本語を含む長いプロジェクトパスを
    毎回埋め込むと超過してしまう。そこで実行内容を1つのbatファイルにまとめ、/TRからは
    そのbatファイルのパス＋race_idだけを渡す（batは自分自身のディレクトリを%~dp0で
    知っているため、プロジェクトパスを繰り返し埋め込む必要がなくなる）。中身は文字コード
    問題を避けるため純粋なASCIIのみで構成する。"""
    content = (
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "set PYTHONIOENCODING=utf-8\r\n"
        f'"{PYTHON_EXE}" predict.py %1 >> "logs\\scheduled\\predict_%1.log" 2>&1\r\n'
    )
    WRAPPER_BAT.write_text(content, encoding="ascii")


def fetch_post_time(race_id: str) -> str:
    """出馬表ページのRaceData01から発走時刻(HH:MM)のみを取得する。取得失敗時は空文字。"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, "html.parser")
        box = soup.find(class_="RaceData01")
        if not box:
            return ""
        m = re.search(r"(\d{1,2}):(\d{2})発走", box.get_text(" ", strip=True))
        return f"{int(m.group(1)):02d}:{m.group(2)}" if m else ""
    except Exception as e:
        logger.warning(f"発走時刻取得に失敗しました (Race ID: {race_id}): {e}")
        return ""


def register_task(race_id: str, trigger_dt: datetime.datetime) -> bool:
    """指定時刻にpredict.pyを1回だけ実行するタスクをタスクスケジューラへ登録する。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    task_name = f"{TASK_PREFIX}{race_id}"
    cmd_line = f'"{WRAPPER_BAT}" {race_id}'
    args = [
        "schtasks", "/Create", "/F",
        "/TN", task_name,
        "/TR", cmd_line,
        "/SC", "ONCE",
        "/SD", trigger_dt.strftime("%Y/%m/%d"),
        "/ST", trigger_dt.strftime("%H:%M"),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"タスク登録失敗 ({task_name}): {result.stderr.strip()}")
        return False
    return True


def cleanup_tasks() -> None:
    """登録済みのKeibaPredict_*タスクを全て削除する。"""
    result = subprocess.run(
        ["schtasks", "/Query", "/FO", "CSV", "/NH"], capture_output=True, text=True
    )
    names = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        first_field = line.split('","')[0].strip('"')
        if TASK_PREFIX in first_field:
            names.append(first_field)

    if not names:
        print("削除対象のタスクはありません。")
        return
    for tn in sorted(set(names)):
        subprocess.run(["schtasks", "/Delete", "/TN", tn, "/F"], capture_output=True, text=True)
        print(f"削除: {tn}")


def main() -> None:
    parser = argparse.ArgumentParser(description="当日レースの発走直前自動予想タスクを登録")
    parser.add_argument("--margin", type=int, default=15, help="発走時刻の何分前に実行するか（既定: 15分）")
    parser.add_argument("--dry-run", action="store_true", help="登録せず、スケジュール内容の確認のみ行う")
    parser.add_argument("--cleanup", action="store_true", help="登録済みのKeibaPredict_*タスクを全て削除して終了")
    args = parser.parse_args()

    if args.cleanup:
        cleanup_tasks()
        return

    if not args.dry_run:
        ensure_wrapper_bat()

    races = RaceScheduleScraper.get_race_ids_by_rounds(target_rounds=None)
    if not races:
        print("本日のレースが見つかりませんでした。")
        return

    now = datetime.datetime.now()
    today = now.date()

    print(f"{'R':<5}{'レース名':<16}{'発走':<7}{'実行予定':<7}{'状態'}")

    scheduled = 0
    skipped = 0
    for r in races:
        post_time_str = fetch_post_time(r["race_id"])
        time.sleep(1.0)

        if not post_time_str:
            print(f"{r['round']:<5}{r['race_name']:<16}{'-':<7}{'-':<7}発走時刻取得失敗")
            skipped += 1
            continue

        h, m = map(int, post_time_str.split(":"))
        post_dt = datetime.datetime.combine(today, datetime.time(h, m))
        trigger_dt = post_dt - datetime.timedelta(minutes=args.margin)

        if trigger_dt <= now:
            print(f"{r['round']:<5}{r['race_name']:<16}{post_time_str:<7}{'-':<7}実行予定時刻超過のためスキップ（手動実行してください）")
            skipped += 1
            continue

        trig_str = trigger_dt.strftime("%H:%M")
        if args.dry_run:
            print(f"{r['round']:<5}{r['race_name']:<16}{post_time_str:<7}{trig_str:<7}(dry-run・未登録)")
            continue

        if register_task(r["race_id"], trigger_dt):
            print(f"{r['round']:<5}{r['race_name']:<16}{post_time_str:<7}{trig_str:<7}登録完了")
            scheduled += 1
        else:
            print(f"{r['round']:<5}{r['race_name']:<16}{post_time_str:<7}{trig_str:<7}登録失敗")
            skipped += 1

    if not args.dry_run:
        print(f"\n{scheduled} 件のタスクを登録しました（スキップ {skipped} 件）。")
        print(f"各タスクの実行ログ: {LOG_DIR}\\predict_<race_id>.log")
        print("終了後・やり直したい場合の一括削除: python schedule_today_predictions.py --cleanup")


if __name__ == "__main__":
    main()
