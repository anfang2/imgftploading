#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_test.py
──────────────────────────────────────────────────────────────
 실제 Watchdog 없이 임시 폴더에 이미지 파일을 '복사'하는 방식으로
 블러 판정 → CSV 기록 → DL 복원(Passthrough) 전체 파이프라인을 검증.

 !! 절대원칙: 이미지 파일 이름 변경 없음 !!
──────────────────────────────────────────────────────────────
 실행 방법:
   python pipeline_test.py
──────────────────────────────────────────────────────────────
"""

import os
import sys
import shutil
import tempfile
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# ─── 같은 폴더의 pipeline_main 에서 공통 로직 임포트 ──────────────
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_main import (
    load_config, get_cfg, setup_logger,
    CSVLogger, compute_blur_score, wait_for_stable,
    PassthroughModel, ImageProcessor
)


# ─────────────────────────────────────────────────────────────────
#  테스트용 이미지 생성 (blur 정도 조절 가능)
# ─────────────────────────────────────────────────────────────────
def make_sharp_image(path: str, size=(512, 512)):
    """선명한 체커보드 패턴 이미지 생성."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cell = 32
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2 == 0:
                img[y:y+cell, x:x+cell] = (220, 220, 220)
            else:
                img[y:y+cell, x:x+cell] = (30, 60, 120)
    cv2.imwrite(path, img)
    return path


def make_blurred_image(path: str, size=(512, 512), ksize=45):
    """인위적으로 흐리게 만든 이미지 생성."""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cell = 32
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2 == 0:
                img[y:y+cell, x:x+cell] = (220, 220, 220)
            else:
                img[y:y+cell, x:x+cell] = (30, 60, 120)
    # 가우시안 블러 (커널 클수록 더 흐림)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    cv2.imwrite(path, blurred)
    return path


# ─────────────────────────────────────────────────────────────────
#  메인 테스트 루틴
# ─────────────────────────────────────────────────────────────────
def run_test():
    setup_logger("DEBUG")
    logging.info("=" * 60)
    logging.info(" Pipeline 단위 테스트 시작")
    logging.info("=" * 60)

    # ── 임시 작업 디렉터리 구성 ──────────────────────────────────
    base_tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    watch_dir  = base_tmp / "input"
    blur_dir   = base_tmp / "blur_classified"
    dl_dir     = base_tmp / "dl_result"
    log_dir    = base_tmp / "logs"
    watch_dir.mkdir(); blur_dir.mkdir(); dl_dir.mkdir(); log_dir.mkdir()

    logging.info(f"[Test] 임시 디렉터리: {base_tmp}")

    # ── 테스트 이미지 생성 ────────────────────────────────────────
    sharp_name  = "CAM01_FRAME_0042.bmp"   # ← 파일명 변경 없이 유지됨을 확인
    blurred_name = "CAM01_FRAME_0099.bmp"

    sharp_path   = str(watch_dir / sharp_name)
    blurred_path = str(watch_dir / blurred_name)

    make_sharp_image(sharp_path)
    make_blurred_image(blurred_path, ksize=55)

    sharp_score   = compute_blur_score(sharp_path)
    blurred_score = compute_blur_score(blurred_path)

    logging.info(f"[Blur Score] 선명 이미지 : {sharp_score:.2f}")
    logging.info(f"[Blur Score] 블러 이미지 : {blurred_score:.2f}")

    # ── 임계값 자동 제안 (테스트에서는 중간값 사용) ───────────────
    threshold = (sharp_score + blurred_score) / 2
    logging.info(f"[Threshold] 자동 권장값: {threshold:.2f} (설정에서 변경 가능)")

    # ── 더미 설정 dict 구성 ──────────────────────────────────────
    cfg_test = {
        "watch_folder"      : str(watch_dir),
        "blur_output_folder": str(blur_dir),
        "dl_output_folder"  : str(dl_dir),
        "blur_threshold"    : threshold,
        "blur_csv_path"     : str(log_dir / "blur_results.csv"),
        "ftp_csv_path"      : str(log_dir / "ftp_results.csv"),
        "file_stable_wait_sec": 1.0,
        "ftp_host"          : "",   # FTP 비활성화 (테스트)
        "model"             : "none"
    }

    # ── CSV 로거 초기화 ───────────────────────────────────────────
    blur_csv = CSVLogger(
        filepath=cfg_test["blur_csv_path"],
        headers=["Date", "Image_Folder", "Blur_Score", "Threshold", "Filename", "Status"]
    )
    ftp_csv = CSVLogger(
        filepath=cfg_test["ftp_csv_path"],
        headers=["Date", "Filename", "DL_Folder", "FTP_Status"]
    )

    # ── Passthrough 모델로 프로세서 생성 ─────────────────────────
    model = PassthroughModel()
    processor = ImageProcessor(cfg_test, model, blur_csv, ftp_csv)

    # ── 두 이미지 처리 ────────────────────────────────────────────
    logging.info("\n── 선명 이미지 처리 ──")
    processor.handle(sharp_path)

    logging.info("\n── 블러 이미지 처리 ──")
    processor.handle(blurred_path)

    # ── 결과 검증 ─────────────────────────────────────────────────
    logging.info("\n" + "=" * 60)
    logging.info(" 결과 검증")
    logging.info("=" * 60)

    # 1) 파일명 변경 없음 확인
    errors = []

    # blur_classified 에는 blurred_name만 있어야 함
    blur_files = list(blur_dir.iterdir())
    if any(f.name == blurred_name for f in blur_files):
        logging.info(f"[PASS] blur_classified 복사 확인: {blurred_name}")
    else:
        logging.error(f"[FAIL] blur_classified에 '{blurred_name}' 없음")
        errors.append("blur copy missing")

    if any(f.name == sharp_name for f in blur_files):
        logging.error(f"[FAIL] 선명 이미지가 blur_classified에 잘못 복사됨: {sharp_name}")
        errors.append("sharp wrongly classified")
    else:
        logging.info(f"[PASS] 선명 이미지는 blur_classified에 없음 (정상)")

    # 2) DL 결과 확인
    dl_files = list(dl_dir.iterdir())
    if any(f.name == blurred_name for f in dl_files):
        logging.info(f"[PASS] DL 결과 저장 확인: {blurred_name}  ← 파일명 변경 없음!")
    else:
        logging.error(f"[FAIL] DL 결과에 '{blurred_name}' 없음")
        errors.append("dl result missing")

    # 3) CSV 확인
    blur_csv_path = Path(cfg_test["blur_csv_path"])
    ftp_csv_path  = Path(cfg_test["ftp_csv_path"])

    if blur_csv_path.exists():
        content = blur_csv_path.read_text(encoding="utf-8-sig")
        logging.info(f"\n[CSV] 블러 분류 결과:\n{content}")
    else:
        logging.error("[FAIL] blur_results.csv 생성 실패")
        errors.append("blur csv missing")

    if ftp_csv_path.exists():
        content = ftp_csv_path.read_text(encoding="utf-8-sig")
        logging.info(f"[CSV] FTP 결과:\n{content}")
    else:
        logging.error("[FAIL] ftp_results.csv 생성 실패")
        errors.append("ftp csv missing")

    # ── 최종 결과 ─────────────────────────────────────────────────
    logging.info("=" * 60)
    if not errors:
        logging.info(" [PASS] 모든 테스트 통과!")
    else:
        logging.error(f" [FAIL] 실패 항목: {errors}")
    logging.info(f" 임시 결과 위치: {base_tmp}")
    logging.info("=" * 60)

    return len(errors) == 0


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
