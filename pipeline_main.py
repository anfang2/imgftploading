#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Windows CP949 콘솔 -> UTF-8 강제 전환
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
========================================================
 Vision Inspection & Deblurring Pipeline  v1.0
 Context: image_pipeline_context_v6.md
 Python 3.12 | torch 2.14 | opencv 5.0 | watchdog 6.0
========================================================

 !! 절대원칙: 이미지 파일 이름은 절대 변경 금지 !!

 Flow:
   Watchdog 감시 → 블러 판정 → X(블러) 복사 → CSV 기록
                              → 딥러닝 복원 → DL결과 저장
                              → FTP 업로드 → FTP결과 CSV
"""

import os
import sys
import json
import time
import shutil
import ftplib
import logging
import threading
import csv
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np

# ─────────────────────────────────────────────
#  Watchdog import (오프라인 wheel 설치 후 사용)
# ─────────────────────────────────────────────
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
except ImportError:
    print("[!] watchdog 패키지가 없습니다. 다음 명령으로 설치하세요:")
    print("    pip install --no-index --find-links=./offline_wheels watchdog")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
#  1. 설정 로더
# ─────────────────────────────────────────────────────────────────
CONFIG_FILENAME = "pipeline_config.json"

def load_config() -> dict:
    """pipeline_config.json 로드. 없으면 기본값 반환."""
    config_path = Path(CONFIG_FILENAME)
    if not config_path.exists():
        logging.warning(f"[Config] {CONFIG_FILENAME} 없음. 기본값 사용.")
        return {}
    with open(config_path, encoding="utf-8") as f:
        raw = f.read()
    # JSON 코멘트(_로 시작하는 키) 제거
    data = json.loads(raw)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def get_cfg(cfg: dict, key: str, default):
    """설정값 취득 (없으면 default)."""
    return cfg.get(key, default)


# ─────────────────────────────────────────────────────────────────
#  2. 로거 설정
# ─────────────────────────────────────────────────────────────────
def setup_logger(log_level: str = "INFO"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)-7s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        handlers=[
                            logging.StreamHandler(sys.stdout),
                            logging.FileHandler("pipeline.log", encoding="utf-8")
                        ])


# ─────────────────────────────────────────────────────────────────
#  3. CSV 로거 (스레드 안전)
# ─────────────────────────────────────────────────────────────────
class CSVLogger:
    """스레드-세이프 CSV 로거."""

    def __init__(self, filepath: str, headers: list):
        self.filepath = Path(filepath)
        self.headers = headers
        self._lock = threading.Lock()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # 파일 없으면 헤더 작성
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(headers)

    def append(self, row: list):
        with self._lock:
            with open(self.filepath, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(row)


# ─────────────────────────────────────────────────────────────────
#  4. 블러 감지 (라플라시안 분산)
# ─────────────────────────────────────────────────────────────────
def compute_blur_score(image_path: str) -> float:
    """
    라플라시안 분산으로 블러 점수 계산.
    낮을수록 더 흐림 (0 = 완전 흐림, 수백 = 선명)
    """
    img = cv2.imread(image_path)
    if img is None:
        return -1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return score


# ─────────────────────────────────────────────────────────────────
#  5. 파일 안정화 대기 (카메라가 쓰는 도중 읽기 방지)
# ─────────────────────────────────────────────────────────────────
def wait_for_stable(filepath: str, timeout: float = 5.0, interval: float = 0.3) -> bool:
    """
    파일 크기가 변화 없이 안정될 때까지 대기.
    Returns True if stable, False if timed out.
    """
    prev_size = -1
    stable_count = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            curr = os.path.getsize(filepath)
            if curr == prev_size and curr > 0:
                stable_count += 1
                if stable_count >= 2:
                    return True
            else:
                stable_count = 0
            prev_size = curr
        except OSError:
            pass
        time.sleep(interval)
    return False


# ─────────────────────────────────────────────────────────────────
#  6. 딥러닝 모델 인터페이스
# ─────────────────────────────────────────────────────────────────
class BaseModel:
    """모든 DL 모델의 기본 인터페이스."""

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            img_bgr: OpenCV BGR 이미지 (np.ndarray, uint8)
        Returns:
            복원된 이미지 (np.ndarray, uint8)
        """
        raise NotImplementedError


class PassthroughModel(BaseModel):
    """모델 없음 -- 원본 그대로 복사 (model: 'none')."""

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        return img_bgr


class NAFNetModel(BaseModel):
    """
    NAFNet 모델 래퍼.
    models/NAFNet 폴더가 sys.path에 있어야 함.
    """

    def __init__(self, weights_path: str, source_dir: str):
        import torch
        if source_dir and os.path.isdir(source_dir):
            if source_dir not in sys.path:
                sys.path.insert(0, source_dir)
        try:
            # NAFNet 공통 import 경로 (NAFNet-main 내부 구조에 따라 조정)
            from basicsr.models.archs.NAFNet_arch import NAFNet
        except ImportError:
            logging.error("[NAFNet] basicsr 모듈을 찾을 수 없습니다. source_dir 경로 확인 필요.")
            raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = self._safe_load(weights_path)
        # NAFNet width32 기본 파라미터
        self.model = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                            enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])
        state = checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)
        logging.info(f"[NAFNet] 모델 로드 완료 ({self.device})")

    def _safe_load(self, path):
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        import torch
        inp = torch.from_numpy(img_bgr[:, :, ::-1].copy()).float().permute(2,0,1).unsqueeze(0) / 255.0
        inp = inp.to(self.device)
        with torch.no_grad():
            out = self.model(inp).squeeze(0).clamp(0,1).cpu().permute(1,2,0).numpy()
        result = (out[:, :, ::-1] * 255).astype(np.uint8)
        return result


class RestormerModel(BaseModel):
    """
    Restormer 모델 래퍼.
    """

    def __init__(self, weights_path: str, source_dir: str):
        import torch
        if source_dir and os.path.isdir(source_dir):
            if source_dir not in sys.path:
                sys.path.insert(0, source_dir)
        try:
            from basicsr.models.archs.restormer_arch import Restormer
        except ImportError:
            logging.error("[Restormer] restormer_arch를 찾을 수 없습니다. source_dir 경로 확인 필요.")
            raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("params", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        self.model = Restormer()
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)
        logging.info(f"[Restormer] 모델 로드 완료 ({self.device})")

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        import torch
        inp = torch.from_numpy(img_bgr[:, :, ::-1].copy()).float().permute(2,0,1).unsqueeze(0) / 255.0
        inp = inp.to(self.device)
        with torch.no_grad():
            out = self.model(inp).squeeze(0).clamp(0,1).cpu().permute(1,2,0).numpy()
        result = (out[:, :, ::-1] * 255).astype(np.uint8)
        return result


class EGRestormerModel(BaseModel):
    """EG-Restormer(ADNet) 모델 래퍼."""

    def __init__(self, weights_path: str, source_dir: str):
        import torch
        if source_dir and os.path.isdir(source_dir):
            if source_dir not in sys.path:
                sys.path.insert(0, source_dir)
        try:
            # ADNet 내부 모듈명 - 실제 경로에 따라 수정 필요
            from basicsr.models.archs.restormer_arch import Restormer as EGRestormer
        except ImportError:
            logging.error("[EGRestormer] 모듈을 찾을 수 없습니다. source_dir 경로 확인 필요.")
            raise

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("params", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        self.model = EGRestormer()
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)
        logging.info(f"[EGRestormer] 모델 로드 완료 ({self.device})")

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        import torch
        inp = torch.from_numpy(img_bgr[:, :, ::-1].copy()).float().permute(2,0,1).unsqueeze(0) / 255.0
        inp = inp.to(self.device)
        with torch.no_grad():
            out = self.model(inp).squeeze(0).clamp(0,1).cpu().permute(1,2,0).numpy()
        result = (out[:, :, ::-1] * 255).astype(np.uint8)
        return result


def build_model(cfg: dict) -> BaseModel:
    """설정에서 모델 선택 후 빌드."""
    name = get_cfg(cfg, "model", "none").lower()
    logging.info(f"[Model] 선택 모델: '{name}'")

    if name == "nafnet":
        return NAFNetModel(
            weights_path=get_cfg(cfg, "nafnet_weights", "weights/NAFNet-GoPro-width32.pth"),
            source_dir=get_cfg(cfg, "nafnet_source_dir", "models/NAFNet")
        )
    elif name == "restormer":
        return RestormerModel(
            weights_path=get_cfg(cfg, "restormer_weights", "weights/motion_deblurring.pth"),
            source_dir=get_cfg(cfg, "restormer_source_dir", "models/Restormer")
        )
    elif name == "eg_restormer":
        return EGRestormerModel(
            weights_path=get_cfg(cfg, "eg_restormer_weights", "weights/egrestormer_gopro.pth"),
            source_dir=get_cfg(cfg, "eg_restormer_source_dir", "models/EG_Restormer")
        )
    else:
        logging.info("[Model] 모델 없음 (Passthrough) - 원본 이미지 그대로 저장")
        return PassthroughModel()


# ─────────────────────────────────────────────────────────────────
#  7. FTP 업로더
# ─────────────────────────────────────────────────────────────────
class FTPUploader:
    """재시도 로직이 포함된 FTP 업로더."""

    def __init__(self, host: str, port: int, user: str, password: str,
                 remote_dir: str, timeout: int = 30, retry: int = 3):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.remote_dir = remote_dir
        self.timeout = timeout
        self.retry = retry

    def upload(self, local_path: str) -> bool:
        """
        local_path의 파일을 FTP remote_dir에 업로드.
        !! 파일명 변경 없음 !!
        Returns True on success.
        """
        filename = Path(local_path).name  # 파일명 그대로 사용
        for attempt in range(1, self.retry + 1):
            try:
                with ftplib.FTP(timeout=self.timeout) as ftp:
                    ftp.connect(self.host, self.port)
                    ftp.login(self.user, self.password)
                    # 원격 디렉토리 이동 (없으면 생성)
                    self._ensure_remote_dir(ftp, self.remote_dir)
                    with open(local_path, "rb") as f:
                        ftp.storbinary(f"STOR {filename}", f)
                logging.info(f"[FTP] 업로드 성공: {filename}")
                return True
            except Exception as e:
                logging.warning(f"[FTP] 시도 {attempt}/{self.retry} 실패: {e}")
                time.sleep(2 * attempt)
        logging.error(f"[FTP] 최종 업로드 실패: {filename}")
        return False

    @staticmethod
    def _ensure_remote_dir(ftp: ftplib.FTP, remote_dir: str):
        """원격 폴더가 없으면 재귀 생성."""
        parts = [p for p in remote_dir.replace("\\", "/").split("/") if p]
        ftp.cwd("/")
        for part in parts:
            try:
                ftp.cwd(part)
            except ftplib.error_perm:
                ftp.mkd(part)
                ftp.cwd(part)


# ─────────────────────────────────────────────────────────────────
#  8. 이미지 처리 워커
# ─────────────────────────────────────────────────────────────────
class ImageProcessor:
    """
    블러 판정 → X복사 → DL 복원 → FTP 업로드 처리 클래스.
    """

    def __init__(self, cfg: dict, model: BaseModel,
                 blur_csv: CSVLogger, ftp_csv: CSVLogger):
        self.cfg = cfg
        self.model = model
        self.blur_csv = blur_csv
        self.ftp_csv = ftp_csv

        self.threshold     = float(get_cfg(cfg, "blur_threshold", 80.0))
        self.blur_out      = Path(get_cfg(cfg, "blur_output_folder", "output/blur_classified"))
        self.dl_out        = Path(get_cfg(cfg, "dl_output_folder", "output/dl_result"))
        self.stable_wait   = float(get_cfg(cfg, "file_stable_wait_sec", 2.0))

        self.blur_out.mkdir(parents=True, exist_ok=True)
        self.dl_out.mkdir(parents=True, exist_ok=True)

        # FTP 설정이 비어있으면 None
        ftp_host = get_cfg(cfg, "ftp_host", "").strip()
        self.ftp: FTPUploader | None = None
        if ftp_host:
            self.ftp = FTPUploader(
                host=ftp_host,
                port=int(get_cfg(cfg, "ftp_port", 21)),
                user=get_cfg(cfg, "ftp_user", ""),
                password=get_cfg(cfg, "ftp_password", ""),
                remote_dir=get_cfg(cfg, "ftp_remote_dir", "/upload"),
                timeout=int(get_cfg(cfg, "ftp_timeout_sec", 30)),
                retry=int(get_cfg(cfg, "ftp_retry_count", 3))
            )
        else:
            logging.warning("[FTP] ftp_host 미설정 -- FTP 업로드 비활성화")

    def handle(self, filepath: str):
        """
        신규 이미지 파일 1장을 처리하는 메인 로직.
        !! 파일명 절대 변경 없음 !!
        """
        src = Path(filepath)
        filename   = src.name              # 원본 파일명 (변경 금지)
        src_folder = str(src.parent)
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── 8-1. 파일 완전히 쓰여질 때까지 대기 ──────────────────
        if not wait_for_stable(filepath, timeout=self.stable_wait):
            logging.warning(f"[Skip] 파일 안정화 실패 (쓰기 중?): {src.name}")
            return

        # ── 8-2. 블러 점수 계산 ───────────────────────────────────
        score = compute_blur_score(filepath)
        if score < 0:
            logging.error(f"[Skip] 이미지 읽기 실패: {src.name}")
            return

        is_blurred = score < self.threshold
        status     = "X" if is_blurred else "OK"
        logging.info(
            f"[Blur] {src.name} | Score={score:.2f} | Threshold={self.threshold} | {status}"
        )

        # ── 8-3. 블러 CSV 기록 ────────────────────────────────────
        # Header: Date, Image_Folder, Blur_Score, Threshold, Filename, Status
        self.blur_csv.append([
            timestamp,
            src_folder,
            f"{score:.4f}",
            f"{self.threshold:.4f}",
            filename,
            status
        ])

        # ── 8-4. X(블러) 이미지 → blur_output_folder 복사 ─────────
        if is_blurred:
            dst_blur = self.blur_out / filename   # 파일명 그대로!
            try:
                shutil.copy2(filepath, dst_blur)  # 메타데이터 포함 복사
                logging.info(f"[Copy] 블러 이미지 복사: {dst_blur}")
            except Exception as e:
                logging.error(f"[Copy] 복사 실패: {e}")
                return

            # ── 8-5. 딥러닝 복원 ──────────────────────────────────
            img_bgr = cv2.imread(str(dst_blur))
            if img_bgr is None:
                logging.error(f"[DL] 이미지 로드 실패: {dst_blur}")
                return

            try:
                restored = self.model.process(img_bgr)
            except Exception as e:
                logging.error(f"[DL] 복원 실패: {e}")
                return

            # ── 8-6. DL 결과 저장 (파일명 그대로!) ──────────────────
            dl_save_path = self.dl_out / filename  # 파일명 절대 변경 없음!
            ext = src.suffix.lower()

            # BMP/PNG/JPEG 등 원본 포맷 유지
            encode_params = []
            if ext in (".jpg", ".jpeg"):
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            elif ext == ".png":
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

            success_write = cv2.imwrite(str(dl_save_path), restored, encode_params)
            if not success_write:
                logging.error(f"[DL] 저장 실패: {dl_save_path}")
                return
            logging.info(f"[DL] 복원 이미지 저장: {dl_save_path}")

            # ── 8-7. FTP 업로드 ───────────────────────────────────
            dl_folder_name = self.dl_out.name
            ftp_ok = False
            if self.ftp:
                ftp_ok = self.ftp.upload(str(dl_save_path))
            else:
                logging.info("[FTP] 비활성화 상태 -- 건너뜀")

            ftp_status = "OK" if ftp_ok else ("NG" if self.ftp else "SKIP")

            # ── 8-8. FTP 결과 CSV 기록 ───────────────────────────
            # Header: Date, Filename, DL_Folder, FTP_Status
            self.ftp_csv.append([
                timestamp,
                filename,
                dl_folder_name,
                ftp_status
            ])


# ─────────────────────────────────────────────────────────────────
#  9. Watchdog 이벤트 핸들러
# ─────────────────────────────────────────────────────────────────
class ImageEventHandler(FileSystemEventHandler):
    """새 이미지 파일 감지 → Queue에 경로 전달."""

    def __init__(self, queue: Queue, extensions: list):
        super().__init__()
        self.queue = queue
        self.extensions = [e.lower() for e in extensions]

    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if Path(path).suffix.lower() in self.extensions:
            logging.debug(f"[Watch] 새 파일 감지: {path}")
            self.queue.put(path)


# ─────────────────────────────────────────────────────────────────
#  10. 워커 스레드
# ─────────────────────────────────────────────────────────────────
def worker_thread(queue: Queue, processor: ImageProcessor, stop_event: threading.Event):
    """Queue에서 파일 경로를 꺼내 처리."""
    logging.info("[Worker] 시작")
    while not stop_event.is_set():
        try:
            filepath = queue.get(timeout=1.0)
            try:
                processor.handle(filepath)
            except Exception as e:
                logging.exception(f"[Worker] 처리 중 예외: {e}")
            finally:
                queue.task_done()
        except Empty:
            continue
    logging.info("[Worker] 종료")


# ─────────────────────────────────────────────────────────────────
#  11. 메인 진입점
# ─────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    setup_logger(get_cfg(cfg, "log_level", "INFO"))

    logging.info("=" * 60)
    logging.info(" Vision Inspection & Deblurring Pipeline v1.0")
    logging.info("=" * 60)

    # ── 감시 폴더 유효성 확인 ─────────────────────────────────────
    watch_folder = get_cfg(cfg, "watch_folder", "")
    if not watch_folder or not os.path.isdir(watch_folder):
        logging.error(f"[Config] watch_folder가 유효하지 않습니다: '{watch_folder}'")
        logging.error("         pipeline_config.json 에서 watch_folder 경로를 설정해 주세요.")
        sys.exit(1)

    extensions = get_cfg(cfg, "image_extensions", [".bmp", ".png", ".jpg", ".jpeg"])
    logging.info(f"[Watch] 감시 폴더: {watch_folder}")
    logging.info(f"[Watch] 감시 확장자: {extensions}")
    logging.info(f"[Blur] 임계값: {get_cfg(cfg, 'blur_threshold', 80.0)}")

    # ── CSV 로거 초기화 ───────────────────────────────────────────
    blur_csv = CSVLogger(
        filepath=get_cfg(cfg, "blur_csv_path", "logs/blur_results.csv"),
        headers=["Date", "Image_Folder", "Blur_Score", "Threshold", "Filename", "Status"]
    )
    ftp_csv = CSVLogger(
        filepath=get_cfg(cfg, "ftp_csv_path", "logs/ftp_results.csv"),
        headers=["Date", "Filename", "DL_Folder", "FTP_Status"]
    )

    # ── 딥러닝 모델 로드 (1회) ────────────────────────────────────
    try:
        model = build_model(cfg)
    except Exception as e:
        logging.error(f"[Model] 모델 로드 실패: {e}")
        logging.error("        모델 없이 실행하려면 config의 'model'을 'none'으로 설정하세요.")
        sys.exit(1)

    # ── 이미지 처리기 & 큐 초기화 ────────────────────────────────
    processor = ImageProcessor(cfg, model, blur_csv, ftp_csv)
    file_queue: Queue = Queue()
    stop_event = threading.Event()

    # ── 워커 스레드 시작 ──────────────────────────────────────────
    n_workers = int(get_cfg(cfg, "worker_threads", 2))
    workers = []
    for i in range(n_workers):
        t = threading.Thread(
            target=worker_thread,
            args=(file_queue, processor, stop_event),
            name=f"Worker-{i+1}",
            daemon=True
        )
        t.start()
        workers.append(t)
    logging.info(f"[Worker] {n_workers}개 워커 스레드 시작")

    # ── Watchdog 시작 (재귀 감시: 최하위 폴더 포함) ──────────────
    event_handler = ImageEventHandler(file_queue, extensions)
    observer = Observer()
    observer.schedule(event_handler, watch_folder, recursive=True)
    observer.start()
    logging.info(f"[Watch] Watchdog 시작 (재귀 감시)")
    logging.info("[Main] 실행 중... 종료하려면 Ctrl+C")

    # ── 메인 루프 (Ctrl+C 대기) ───────────────────────────────────
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("[Main] 종료 신호 수신")

    # ── 종료 처리 ─────────────────────────────────────────────────
    observer.stop()
    observer.join()
    stop_event.set()
    for t in workers:
        t.join(timeout=10)
    logging.info("[Main] 파이프라인 종료 완료.")


if __name__ == "__main__":
    main()
