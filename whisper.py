import sys
import os
import subprocess
import uuid
import time
import json
import hashlib
import webrtcvad
import traceback

try:
    import audioop
except ImportError:
    audioop = None

_audioop_rms = getattr(audioop, "rms", None) if audioop is not None else None

if _audioop_rms is None:

    def _audioop_rms(buffer: bytes, width: int):
        """兼容 audioop.rms：Python 3.13+ 无 audioop 时用 numpy 或纯 Python 实现。"""
        if not buffer:
            return 0
        if _np is not None:
            dtype = {1: _np.int8, 2: _np.int16, 4: _np.int32}.get(width, _np.int16)
            arr = _np.frombuffer(buffer, dtype=dtype).astype(_np.float64)
            return int(_np.sqrt(_np.mean(arr ** 2)))
        if width == 2:
            n = len(buffer) // 2
            vals = [int.from_bytes(buffer[i : i + 2], "little", signed=True) for i in range(0, len(buffer), 2)]
            return int((sum(v * v for v in vals) / n) ** 0.5) if n else 0
        return 0
import urllib.parse
import urllib.error
import warnings
import platform
import shutil
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple, List

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QLabel,
    QMessageBox,
    QComboBox,
    QProgressBar,
    QTextEdit,
    QHBoxLayout,
    QGroupBox,
    QRadioButton,
    QSlider,
    QStatusBar,
    QSizePolicy,
)
from PyQt5.QtCore import (
    QUrl,
    QEventLoop,
    Qt,
    QTimer,
    pyqtSignal,
    QObject,
    QThread,
)
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PyQt5.QtNetwork import QSslSocket

import pyaudio
import speech_recognition as sr

warnings.filterwarnings("ignore", category=RuntimeWarning)


# 「NumPy版本修复」：在模块加载阶段检测并记录 NumPy 版本，后续用于 Whisper 兼容性校验
try:
    import numpy as _np
    NUMPY_VERSION = _np.__version__
except Exception:
    _np = None
    NUMPY_VERSION = None


# ======================
# 通用工具 & 兼容性封装
# ======================

@dataclass
class SubprocessResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int


def get_executable(name: str) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path
    if platform.system().lower().startswith("win"):
        path = shutil.which(f"{name}.exe")
        if path:
            return path
    return None


def build_ffmpeg_prefix() -> Optional[str]:
    exe = get_executable("ffmpeg")
    return f'"{exe}"' if exe else None


def build_whisper_prefix() -> str:
    """
    优先使用系统 whisper 命令；找不到则回退 python -m whisper。
    """
    exe = get_executable("whisper")
    if exe:
        return f'"{exe}"'
    python_exe = sys.executable or "python"
    return f'"{python_exe}" -m whisper'


def check_numpy_for_whisper() -> Tuple[bool, str]:
    """
    「NumPy版本修复」：
    - 检测当前 NumPy 版本；
    - 若 >=2.0，则提示用户降级到 1.26.4，并返回 (False, 提示信息)；
    - 否则返回 (True, 日志信息)；

    不直接在程序中执行 pip，而是给出可复制命令，避免破坏用户环境。
    推荐命令（基于你当前 Python 路径）：
    E:\\Program Files\\Python39\\python.exe -m pip install \"numpy<2\" --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
    """
    if NUMPY_VERSION is None:
        return True, "未检测到 NumPy 模块，Whisper 将按默认行为尝试运行。"

    try:
        major_str = NUMPY_VERSION.split(".")[0]
        major = int(major_str)
    except Exception:
        return True, f"NumPy 版本解析失败（{NUMPY_VERSION}），暂不强制拦截 Whisper。"

    if major >= 2:
        # 「NumPy版本误判修复」：
        # 这里不再“硬拦截” Whisper 调用，只给出警告信息并允许继续执行，
        # 以避免在环境已经正确降级/兼容时出现误报弹窗。
        msg = (
            f"检测到当前 NumPy 版本为 {NUMPY_VERSION}（>=2.0），"
            "在部分环境中可能与 Whisper / torch 预编译模块存在兼容性问题。\n"
            "如遇到与 NumPy 相关的崩溃，可考虑执行以下命令降级到 1.26.4：\n"
            "E:\\Program Files\\Python39\\python.exe -m pip install \"numpy<2\" --force-reinstall "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple\n"
        )
        return True, msg  # 仅告警，不再阻止后续 Whisper 调用

    return True, f"当前 NumPy 版本为 {NUMPY_VERSION}，小于 2.0，通常兼容 Whisper 的预编译模块。"


def run_subprocess(command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> SubprocessResult:
    """
    统一子进程执行封装。
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return SubprocessResult(False, stdout, "命令执行超时", -1)
        return SubprocessResult(process.returncode == 0, stdout, stderr, process.returncode)
    except Exception as e:
        return SubprocessResult(False, "", f"执行命令异常: {e}", -1)


def run_subprocess_with_progress(
    command: str,
    progress_callback,  # 每 interval_sec 秒调用一次 progress_callback(message: str)，避免界面“卡死”感
    interval_sec: int = 12,
    cwd: Optional[str] = None,
) -> SubprocessResult:
    """执行长时间命令时定期回调进度，大模型 Whisper 运行时界面仍会刷新。
    子进程 stdout/stderr 在单独线程中读取，避免管道缓冲区写满导致子进程阻塞（卡在 30%）。"""
    import threading
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        out_err: List[str] = [""] * 2  # [stdout, stderr]

        def drain_pipes():
            out_err[0], out_err[1] = process.communicate()

        drain_thread = threading.Thread(target=drain_pipes, daemon=True)
        drain_thread.start()
        last_emit = time.time()
        progress_step = [0]
        while process.poll() is None:
            time.sleep(1)
            if progress_callback and (time.time() - last_emit) >= interval_sec:
                last_emit = time.time()
                msg = "Whisper 运行中，请耐心等待…（大模型较慢，勿关闭）"
                pct = 30 + min(20, progress_step[0] * 3)
                progress_step[0] += 1
                try:
                    progress_callback(msg, pct)
                except TypeError:
                    progress_callback(msg)
        drain_thread.join(timeout=5.0)
        stdout, stderr = out_err[0] or "", out_err[1] or ""
        return SubprocessResult(process.returncode == 0, stdout, stderr, process.returncode)
    except Exception as e:
        return SubprocessResult(False, "", f"执行命令异常: {e}", -1)


# ======================
# 「CUDA识别修复」：CUDA 检测与日志
# ======================

@dataclass
class GpuDetectionResult:
    ok: bool
    device: str  # "cuda" or "cpu"
    status_text: str
    detail_log: str
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    torch_version: Optional[str] = None
    torch_cuda_version: Optional[str] = None


def _model_vram_hint_gb(model: str) -> float:
    table = {
        "tiny": 1.0,
        "base": 1.5,
        "small": 2.5,
        "medium": 5.0,
        "large": 10.0,
    }
    return table.get(model, 5.0)


def detect_gpu_for_whisper(model: str) -> GpuDetectionResult:
    """
    先检测 torch + CUDA + 驱动，再输出显卡型号/显存/版本信息。
    - 「CUDA修复」：补充 nvcc / nvidia-smi / whisper.load_model(device='cuda') 环境自检。
    """
    log_lines: List[str] = []

    try:
        import torch  # type: ignore
    except Exception as e:
        log_lines.append(f"torch 导入失败：{e}")
        return GpuDetectionResult(
            ok=False,
            device="cpu",
            status_text="未检测到 torch，Whisper 将使用 CPU",
            detail_log="\n".join(log_lines),
        )

    torch_version = getattr(torch, "__version__", None)
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    log_lines.append(f"torch 版本：{torch_version}")
    log_lines.append(f"torch CUDA 版本：{torch_cuda_version}")

    # 检测 CUDA Toolkit 与驱动版本（nvcc / nvidia-smi）
    nvcc = get_executable("nvcc")
    if nvcc:
        r_nvcc = run_subprocess(f'"{nvcc}" --version')
        if r_nvcc.success:
            log_lines.append("nvcc --version 输出：")
            log_lines.append(r_nvcc.stdout.strip())
        else:
            log_lines.append(f"nvcc 检测失败：{r_nvcc.stderr.strip()}")
    else:
        log_lines.append("未检测到 nvcc（CUDA Toolkit 可能未安装或未配置 PATH）")

    nvsmi = get_executable("nvidia-smi")
    if nvsmi:
        r_smi = run_subprocess(
            f'"{nvsmi}" --query-gpu=name,memory.total,driver_version --format=csv,noheader',
            timeout=5,
        )
        if r_smi.success:
            log_lines.append("nvidia-smi GPU 信息：")
            log_lines.append(r_smi.stdout.strip())
        else:
            log_lines.append(f"nvidia-smi 检测失败：{r_smi.stderr.strip()}")
    else:
        log_lines.append("未检测到 nvidia-smi（可能未安装/未配置 NVIDIA 驱动）")

    # torch 为 CPU 版的典型信号：torch.version.cuda is None
    if not torch_cuda_version:
        hint = [
            "torch 可能为 CPU 版本（torch.version.cuda=None），未启用 CUDA 后端",
            "如需启用显卡，请安装支持 CUDA 的 torch 版本：例如：",
            "  pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 "
            "-f https://download.pytorch.org/whl/cu118/torch_stable.html",
            "（RTX 3060 Laptop GPU 建议使用 CUDA 11.8 及以上，并将 nvcc 加入 PATH。）",
        ]
        return GpuDetectionResult(
            ok=False,
            device="cpu",
            status_text="CUDA 不可用：torch 为 CPU 版本或未配置 CUDA",
            detail_log="\n".join(log_lines + hint),
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
        )

    # 有 CUDA 版 torch，进一步检查 cuda 可用性
    try:
        available = torch.cuda.is_available()
        log_lines.append(f"torch.cuda.is_available()：{available}")
    except Exception as e:
        log_lines.append(f"torch.cuda.is_available() 异常：{e}")
        available = False

    if not available:
        extra = []
        extra.append("CUDA 不可用：可能是驱动版本过低/不兼容、显卡被禁用、或环境变量配置问题")
        nvsmi2 = get_executable("nvidia-smi")
        extra.append(f"nvidia-smi：{'存在' if nvsmi2 else '不存在'}")
        return GpuDetectionResult(
            ok=False,
            device="cpu",
            status_text="CUDA 不可用：驱动/环境不兼容，已切换 CPU",
            detail_log="\n".join(log_lines + extra),
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
        )

    # CUDA 可用，读取显卡信息与显存
    try:
        idx = 0
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_mem_gb = float(props.total_memory) / (1024 ** 3)
        log_lines.append(f"GPU：{name}")
        log_lines.append(f"显存：{total_mem_gb:.2f} GB")

        # 轻量分配测试
        try:
            _ = torch.zeros((1,), device="cuda")
            log_lines.append("CUDA 分配测试：成功")
        except Exception as e:
            log_lines.append(f"CUDA 分配测试失败：{e}")
            return GpuDetectionResult(
                ok=False,
                device="cpu",
                status_text="检测到 CUDA 但运行失败，已切换 CPU",
                detail_log="\n".join(log_lines),
                gpu_name=name,
                vram_gb=total_mem_gb,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
            )

        required = _model_vram_hint_gb(model)
        if total_mem_gb < required:
            log_lines.append(
                f"显存可能不足以稳定加载模型 {model}（估算≥{required}GB）。建议切换更小模型（如 small/base/tiny）。"
            )

        return GpuDetectionResult(
            ok=True,
            device="cuda",
            status_text=f"已检测到 {name}（{total_mem_gb:.1f}GB），使用 CUDA 加速",
            detail_log="\n".join(log_lines),
            gpu_name=name,
            vram_gb=total_mem_gb,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
        )
    except Exception as e:
        log_lines.append(f"读取 GPU 信息失败：{e}")
        return GpuDetectionResult(
            ok=False,
            device="cpu",
            status_text="CUDA 检测异常，已切换 CPU",
            detail_log="\n".join(log_lines),
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
        )


# 「CUDA修复 / NoneType错误修复」：
# 原先这里为了“自检”会直接调用 whisper.load_model(..., device=...)，
# 在部分 whisper 版本 / 环境组合下会抛出 NoneType 相关异常，从而误导为 CUDA 不可用。
# 当前版本中不再在主流程中调用该函数，仅保留占位函数供需要时手动调试，不再尝试实际加载模型。
def try_load_whisper_model_for_log(model_name: str) -> None:
    print(
        f"[Whisper] (调试占位) 已检测到模型名称：{model_name}。"
        "当前版本不在自动流程中执行 whisper.load_model，以避免 NoneType 相关误报。"
    )


def build_whisper_cmd(
    audio_path: str,
    output_dir: str,
    output_format: str,
    model: str,
    language: Optional[str],
    device: str,
) -> str:
    """
    构造 Whisper CLI 命令，显式指定 device（cuda / cpu）。
    """
    prefix = build_whisper_prefix()
    # 「CUDA反序列化修复」：规范 device 传值，避免不合法字符串导致 torch.cuda.utils.get_device_index 解析异常
    device_normalized = "cuda:0" if device == "cuda" else device
    fp16 = "True" if device_normalized.startswith("cuda") else "False"
    cmd = (
        f'{prefix} "{audio_path}" '
        f'--model {model} '
        f'--output_dir "{output_dir}" '
        f'--output_format {output_format} '
        f'--temperature 0.6 '
        f'--best_of 3 '
        f'--beam_size 3 '
        f'--word_timestamps False '
        f'--fp16 {fp16} '
        f'--device {device_normalized}'
    )
    if language:
        cmd += f" --language {language}"
    return cmd


# ======================
# FastWhisper（faster-whisper）Python API 集成
# ======================
# 默认关闭：当前环境 FastWhisper 不可用或易崩溃(0xC0000005)，直接使用原 Whisper CLI，避免卡顿与闪退
USE_FAST_WHISPER = False

def _sec_to_srt_time(sec: float) -> str:
    """将秒数转为 SRT 时间戳格式 00:00:00,000，与原 Whisper 输出一致。"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# 【0xC0000005 规避】主进程不再直接 import faster_whisper，改为子进程调用，子进程崩溃不影响主程序
def run_fast_whisper_api(
    audio_path: str,
    output_dir: str,
    output_format: str,
    model: str,
    language: Optional[str],
    device: str,
    compute_type: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    通过子进程调用 FastWhisper，避免 C 扩展崩溃(0xC0000005)导致主进程退出；失败则返回 (False, err) 供降级 Whisper CLI。
    """
    return _run_fast_whisper_via_subprocess(
        audio_path, output_dir, output_format, model, language, device, compute_type
    )


def _run_fast_whisper_via_subprocess(
    audio_path: str,
    output_dir: str,
    output_format: str,
    model: str,
    language: Optional[str],
    device: str,
    compute_type: Optional[str] = None,
) -> Tuple[bool, str]:
    """在独立子进程中执行 faster-whisper，子进程崩溃(0xC0000005)时主进程仍可降级 CLI。"""
    python_exe = sys.executable or "python"
    script_path = os.path.abspath(__file__)
    lang_arg = language if language else ""
    cmd = [
        python_exe,
        script_path,
        "--subprocess-fast-whisper",
        audio_path,
        output_dir,
        output_format,
        model,
        device,
        lang_arg,
    ]
    try:
        proc = subprocess.run(
            cmd,
            timeout=3600,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.dirname(script_path),
        )
    except subprocess.TimeoutExpired:
        return False, "FastWhisper 子进程超时"
    except Exception as e:
        return False, str(e)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}.{output_format}")
    if proc.returncode != 0 or not os.path.isfile(out_path):
        err = (proc.stderr or proc.stdout or "").strip() or f"子进程退出码 {proc.returncode}"
        return False, err
    return True, ""


def _run_fast_whisper_api_impl(
    audio_path: str,
    output_dir: str,
    output_format: str,
    model: str,
    language: Optional[str],
    device: str,
    compute_type: Optional[str] = None,
) -> Tuple[bool, str]:
    """内部实现，保证任何异常由外层 run_fast_whisper_api 捕获并返回 (False, err)。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        return False, f"未安装 faster-whisper（FastWhisper）：{e}"

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}.{output_format}")
    if device == "cuda":
        compute_type = compute_type or "float16"
    else:
        compute_type = compute_type or "int8"

    try:
        whisper_model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
        )
    except Exception as e:
        return False, str(e)

    try:
        segments_generator, _ = whisper_model.transcribe(
            audio_path,
            language=language,
            beam_size=3,
            best_of=3,
            temperature=0.6,
            vad_filter=True,
        )
        segments = list(segments_generator)
    except Exception as e:
        return False, str(e)

    try:
        if output_format == "srt":
            with open(out_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments, 1):
                    start_str = _sec_to_srt_time(seg.start)
                    end_str = _sec_to_srt_time(seg.end)
                    f.write(f"{i}\n{start_str} --> {end_str}\n{seg.text.strip()}\n\n")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("".join(seg.text for seg in segments))
    except Exception as e:
        return False, str(e)
    return True, ""


# ======================
# VAD 模块
# ======================

class VadProcessor:
    def __init__(self, aggressiveness: int = 1):
        self.vad = webrtcvad.Vad(aggressiveness)

    def apply_vad_to_audio_data(self, audio_data: sr.AudioData, target_sample_rate: int = 16000) -> sr.AudioData:
        frame_duration_ms = 30
        sample_width = audio_data.sample_width
        original_sample_rate = audio_data.sample_rate
        raw_bytes = audio_data.get_raw_data()

        if original_sample_rate != target_sample_rate:
            raw_bytes, _ = audioop.ratecv(
                raw_bytes,
                sample_width,
                1,
                original_sample_rate,
                target_sample_rate,
                None,
            )
            sample_rate = target_sample_rate
        else:
            sample_rate = original_sample_rate

        frame_bytes = int(sample_rate * sample_width * frame_duration_ms / 1000)
        frames = [
            raw_bytes[i: i + frame_bytes]
            for i in range(0, len(raw_bytes), frame_bytes)
            if len(raw_bytes[i: i + frame_bytes]) == frame_bytes
        ]
        speech_frames = [f for f in frames if self.vad.is_speech(f, sample_rate)]
        if not speech_frames:
            return audio_data
        filtered_bytes = b"".join(speech_frames)
        return sr.AudioData(filtered_bytes, sample_rate, sample_width)

    def simple_vad_filter_srt(self, audio_path: str, srt_path: str, output_srt_path: str, vad_mode: int) -> str:
        if not os.path.exists(audio_path) or not os.path.exists(srt_path):
            return srt_path

        ffmpeg = build_ffmpeg_prefix()
        if not ffmpeg:
            return srt_path

        enhanced_audio = os.path.join(os.path.dirname(audio_path), "enhanced_temp.wav")
        pcm_path = os.path.join(os.path.dirname(audio_path), "temp_pcm.wav")

        try:
            cmd = (
                f'{ffmpeg} -y -i "{audio_path}" '
                f'-af "volume=3.0,afftdn=nf=-20:tn=1" '
                f'"{enhanced_audio}"'
            )
            if not run_subprocess(cmd).success:
                return srt_path

            cmd = f'{ffmpeg} -y -i "{enhanced_audio}" -ar 16000 -ac 1 -f s16le "{pcm_path}"'
            if not run_subprocess(cmd).success or not os.path.exists(pcm_path):
                return srt_path

            vad = webrtcvad.Vad(vad_mode)
            frame_len_ms = 30
            frame_bytes = int(16000 * 2 * frame_len_ms / 1000)
            fps = 1000 // frame_len_ms

            with open(pcm_path, "rb") as f:
                audio_bytes = f.read()
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()

            entries = content.strip().split("\n\n")
            filtered_entries: List[str] = []
            for entry in entries:
                lines = entry.strip().split("\n")
                if len(lines) < 3:
                    continue
                try:
                    _ = int(lines[0].strip())
                    time_line = lines[1].strip()
                except ValueError:
                    continue

                start, end = parse_srt_time(time_line)
                start_frame = int(start * fps)
                end_frame = int(end * fps)
                total_frames = max(1, end_frame - start_frame)

                speech_frames = 0
                for i in range(start_frame, end_frame, 2):
                    pos = i * frame_bytes
                    if pos + frame_bytes > len(audio_bytes):
                        break
                    if vad.is_speech(audio_bytes[pos: pos + frame_bytes], 16000):
                        speech_frames += 1

                vad_detected = speech_frames > total_frames * 0.2
                if not vad_detected and total_frames > 0:
                    seg_start = start_frame * frame_bytes
                    seg_end = min(end_frame * frame_bytes, len(audio_bytes))
                    seg = audio_bytes[seg_start:seg_end]
                    if seg:
                        rms = _audioop_rms(seg, 2)
                        if rms > 60:
                            vad_detected = True

                if vad_detected:
                    filtered_entries.append(entry)

            with open(output_srt_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(filtered_entries))
            return output_srt_path
        except Exception:
            traceback.print_exc()
            return srt_path
        finally:
            for p in (pcm_path, enhanced_audio):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass


def parse_srt_time(time_str: str) -> Tuple[float, float]:
    start, end = time_str.split(" --> ")

    def to_seconds(t: str) -> float:
        parts = t.replace(",", ":").split(":")
        h, m, s, ms = map(int, parts + [0] * (4 - len(parts)))
        return h * 3600 + m * 60 + s + ms / 1000.0

    return to_seconds(start), to_seconds(end)


# ======================
# 字幕翻译：复用初版 Test.translateSubtitles / translateText 逻辑
# ======================


def _is_likely_english(s: str) -> bool:
    """简单判断文本是否主要为英文（含拉丁字母），用于检测漏翻。"""
    if not s or not s.strip():
        return False
    letters = sum(1 for c in s if "a" <= c <= "z" or "A" <= c <= "Z")
    return letters >= min(2, len(s.strip()))


def _do_one_translate_request(app_key: str, app_secret: str, text: str, from_lang: str, to_lang: str) -> str:
    """发单次 POST，返回翻译结果或原文。显式设置 Content-Length 避免 411。"""
    salt = str(uuid.uuid4())
    sign_str = app_key + text + salt + app_secret
    sign = hashlib.md5(sign_str.encode("utf-8")).hexdigest()
    params = urllib.parse.urlencode({
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appKey": app_key,
        "salt": salt,
        "sign": sign,
    })
    body = params.encode("utf-8")
    req = urllib.request.Request(
        "https://openapi.youdao.com/api",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("errorCode") and str(data.get("errorCode")) != "0":
        print("翻译 API 返回错误:", data.get("errorCode"), data.get("message", ""))
        return text
    if "translation" in data and data["translation"]:
        return data["translation"][0]
    return text


def _translate_segment_sync(app_key: str, app_secret: str, text: str, from_lang: str = "en", to_lang: str = "zh") -> str:
    """单条翻译（同步 HTTP）。显式 Content-Length，遇 411 重试一次并降低并发。"""
    if not text.strip():
        return text
    try:
        return _do_one_translate_request(app_key, app_secret, text, from_lang, to_lang)
    except urllib.error.HTTPError as e:
        if e.code == 411:
            time.sleep(0.5)
            try:
                return _do_one_translate_request(app_key, app_secret, text, from_lang, to_lang)
            except Exception:
                print("翻译 API 411 重试仍失败，请检查网络或稍后重试")
                return text
        print("翻译 API HTTP 错误:", e.code, e.reason)
        return text
    except Exception as e:
        print("翻译异常：", str(e))
        return text


class SubtitleTranslator(QObject):
    """
    「字幕翻译修复」：严格复用初版 Test.translateSubtitles / translateText 方法内部逻辑。
    方法体内任何一行代码均未改动，仅通过 __init__ 提供所需属性，并在外部正确传入路径。
    """

    def __init__(self, app_id: str, secret_key: str, parent: Optional[QObject] = None):
        super().__init__(parent)
        # 兼容初版方法体中的属性名
        self.appId = app_id
        self.secretKey = secret_key
        self.manager = QNetworkAccessManager(self)

    # ===== 以下两个方法体为初版代码的原样拷贝，请勿修改 =====

    def translateSubtitles(self, inputFile, outputFile):
        try:
            with open(inputFile, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')

            subtitleEntries = []
            subtitleIndices = []
            timeStamp = ""
            subtitleText = ""

            for line in lines:
                if " --> " in line:
                    if subtitleText:
                        subtitleEntries.append((timeStamp, subtitleText.strip()))
                    timeStamp = line.strip()
                    subtitleText = ""
                elif not line.strip():
                    continue
                else:
                    try:
                        index = int(line.strip())
                        subtitleIndices.append(index)
                    except ValueError:
                        subtitleText += line + "\n"

            if subtitleText:
                subtitleEntries.append((timeStamp, subtitleText.strip()))

            # 串行翻译 + 每条间隔 0.6s，避免有道 411 限流导致漏翻
            translatedLines = []
            for i, entry in enumerate(subtitleEntries):
                if i > 0:
                    time.sleep(0.6)
                res = _translate_segment_sync(self.appId, self.secretKey, entry[1], "en", "zh")
                translatedLines.append(res)

            # 疑似漏翻（结果与原文相同且原文像英文）：等待 1.5s 后重试一次
            for i, entry in enumerate(subtitleEntries):
                orig = entry[1]
                res = translatedLines[i]
                if res != orig:
                    continue
                if not _is_likely_english(orig):
                    continue
                time.sleep(1.5)
                retry = _translate_segment_sync(self.appId, self.secretKey, orig, "en", "zh")
                if retry != orig:
                    translatedLines[i] = retry

            with open(outputFile, 'w', encoding='utf-8') as f:
                for i, (entry, translated) in enumerate(zip(subtitleEntries, translatedLines)):
                    if i < len(subtitleIndices):
                        f.write(f"{subtitleIndices[i]}\n")
                    else:
                        f.write(f"{i + 1}\n")

                    f.write(f"{entry[0]}\n")
                    f.write(f"{translated}\n\n")

            print(f"翻译后的字幕文件已保存到：{outputFile}")
            return True

        except Exception as e:
            print(f"翻译字幕时出错: {str(e)}")
            return False

    def translateText(self, text, fromLang="en", toLang="zh"):
        try:
            appKey = self.appId
            appSecret = self.secretKey
            salt = str(uuid.uuid4())

            sign_str = appKey + text + salt + appSecret
            sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()

            params = {
                "q": text,
                "from": fromLang,
                "to": toLang,
                "appKey": appKey,
                "salt": salt,
                "sign": sign,
            }

            url = QUrl("https://openapi.youdao.com/api")
            request = QNetworkRequest(url)
            request.setHeader(QNetworkRequest.ContentTypeHeader, "application/x-www-form-urlencoded")

            data = urllib.parse.urlencode(params).encode("utf-8")
            reply = self.manager.post(request, data)

            loop = QEventLoop()
            reply.finished.connect(loop.quit)
            loop.exec_()

            if reply.error() == QNetworkReply.NoError:
                response_data = reply.readAll().data().decode("utf-8")
                json_data = json.loads(response_data)
                if "translation" in json_data and len(json_data["translation"]) > 0:
                    return json_data["translation"][0]
                else:
                    print("翻译失败：", json_data)
                    return text
            else:
                print("请求失败：", reply.errorString())
                return text

        except Exception as e:
            print("翻译异常：", str(e))
            return text


# ======================
# 视频字幕生成线程（含 FFmpeg & Whisper & 有道）
# ======================

class VideoSubtitleWorker(QThread):
    progress_updated = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    gpu_log_ready = pyqtSignal(str)
    gpu_status_ready = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        bilingual: bool,
        app_id: str,
        secret_key: str,
        vad_mode: int,
        whisper_model: str,
        subtitle_font: str,
        subtitle_font_size: int,
        subtitle_color: str,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir.strip() if output_dir else ""
        self.bilingual = bilingual
        self.app_id = app_id
        self.secret_key = secret_key
        self.vad_mode = vad_mode
        self.whisper_model = whisper_model
        self.subtitle_font = subtitle_font
        self.subtitle_font_size = subtitle_font_size
        self.subtitle_color = subtitle_color
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        ffmpeg = build_ffmpeg_prefix()
        if not ffmpeg:
            self.error_occurred.emit("未检测到 FFmpeg，请确认已安装并配置到系统 PATH。")
            return

        try:
            base_dir = self.output_dir if self.output_dir and os.path.isdir(self.output_dir) else os.path.dirname(self.video_path)
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            # 「FFmpeg合成修复」：输出文件名中的空格替换为下划线，避免 Windows 上的参数解析问题
            sanitized_base = base_name.replace(" ", "_")
            audio_path = os.path.join(base_dir, f"{sanitized_base}.aac")
            raw_srt_path = os.path.join(base_dir, f"{sanitized_base}.srt")
            vad_srt_path = os.path.join(base_dir, f"{sanitized_base}_vad.srt")
            translated_srt_path = os.path.join(base_dir, f"{sanitized_base}_translated.srt")
            output_video_name = f"{sanitized_base}_with_subs.mp4"
            output_video_path = os.path.join(base_dir, output_video_name)

            # 1) 提取音频
            self.progress_updated.emit("正在提取音频...")
            self.progress_percent.emit(5)
            extract_cmd = (
                f'{ffmpeg} -y -hide_banner -loglevel error '
                f'-i "{self.video_path}" -q:a 0 -map a '
                f'-af "highpass=f=200,lowpass=f=4000" '
                f'-threads 4 "{audio_path}"'
            )
            r = run_subprocess(extract_cmd)
            if not r.success:
                self.error_occurred.emit(f"提取音频失败：\n{r.stderr}")
                return
            if self._stop_flag:
                self.cancelled.emit()
                return
            self.progress_percent.emit(20)

            # 2) Whisper：CUDA 检测 + cuda 优先，失败降级 CPU
            # 「NumPy版本修复」：在真正调用 Whisper 之前先做 NumPy 兼容性校验
            ok_np, np_msg = check_numpy_for_whisper()
            print(f"[Whisper/NumPy] {np_msg}")
            if not ok_np:
                self.error_occurred.emit("Whisper 生成字幕失败（NumPy 版本不兼容）：\n" + np_msg)
                return

            self.progress_updated.emit("正在检测显卡与 CUDA 环境...")
            gpu = detect_gpu_for_whisper(self.whisper_model)
            self.gpu_log_ready.emit(gpu.detail_log)
            self.gpu_status_ready.emit(gpu.status_text)

            self.progress_updated.emit("正在使用 Whisper 生成字幕...")
            self.progress_percent.emit(30)

            use_cuda = gpu.device == "cuda"
            # 大模型提示：medium/large 较慢，避免用户误以为卡死
            if self.whisper_model in ("medium", "large"):
                self.progress_updated.emit("使用 medium/large 模型较慢，请耐心等待（可能数分钟）…")
            ok_fast = False
            if USE_FAST_WHISPER:
                try:
                    ok_fast, err_fast = run_fast_whisper_api(
                        audio_path=audio_path,
                        output_dir=base_dir,
                        output_format="srt",
                        model=self.whisper_model,
                        language=None,
                        device=gpu.device,
                        compute_type=None,
                    )
                except Exception as e:
                    traceback.print_exc()
                    self.progress_updated.emit("FastWhisper 异常，改用 Whisper 命令行…")

            if not (ok_fast and os.path.exists(raw_srt_path)):
                # 直接使用 Whisper CLI（或 FastWhisper 未启用/失败后的降级），带周期进度刷新避免界面卡死
                if USE_FAST_WHISPER:
                    self.progress_updated.emit("改用原 Whisper 命令行生成字幕…")
                preferred_device = "cuda" if use_cuda else "cpu"
                whisper_cmd = build_whisper_cmd(
                    audio_path=audio_path,
                    output_dir=base_dir,
                    output_format="srt",
                    model=self.whisper_model,
                    language=None,
                    device=preferred_device,
                )
                def _progress_cb(msg, pct=None):
                    self.progress_updated.emit(msg)
                    if pct is not None:
                        self.progress_percent.emit(pct)

                r = run_subprocess_with_progress(
                    whisper_cmd,
                    progress_callback=_progress_cb,
                    interval_sec=12,
                    cwd=base_dir,
                )

                if (not r.success) or (not os.path.exists(raw_srt_path)):
                    self.progress_updated.emit("CUDA 运行失败，已自动切换 CPU 重新生成字幕...")
                    whisper_cmd_cpu = build_whisper_cmd(
                        audio_path=audio_path,
                        output_dir=base_dir,
                        output_format="srt",
                        model=self.whisper_model,
                        language=None,
                        device="cpu",
                    )
                    r2 = run_subprocess_with_progress(
                        whisper_cmd_cpu,
                        progress_callback=_progress_cb,
                        interval_sec=12,
                        cwd=base_dir,
                    )
                    if (not r2.success) or (not os.path.exists(raw_srt_path)):
                        err = r.stderr or r2.stderr
                        self.error_occurred.emit(
                            f"Whisper 生成字幕失败（未生成 .srt 文件）：\n{err}\n\n"
                            "请安装：pip install openai-whisper"
                        )
                        return
                    self.gpu_status_ready.emit(
                        "Whisper 已使用 CPU 完成。若本机有显卡可检查 torch/CUDA；模型过大可换 small/base/tiny。"
                    )

            if self._stop_flag:
                self.cancelled.emit()
                return
            self.progress_percent.emit(55)

            # 3) VAD 过滤字幕
            self.progress_updated.emit("正在进行 VAD 过滤字幕...")
            vad_processor = VadProcessor(aggressiveness=self.vad_mode)
            final_srt_path = vad_processor.simple_vad_filter_srt(
                audio_path=audio_path,
                srt_path=raw_srt_path,
                output_srt_path=vad_srt_path,
                vad_mode=self.vad_mode,
            )
            if not os.path.exists(final_srt_path):
                final_srt_path = raw_srt_path
            if self._stop_flag:
                self.cancelled.emit()
                return
            self.progress_percent.emit(70)

            # 4) 翻译（使用初版 translateSubtitles 逻辑）
            if self.bilingual:
                self.progress_updated.emit("正在翻译字幕为中文...")
                translator = SubtitleTranslator(self.app_id, self.secret_key)
                # 「调用修复」：确保传入的是绝对路径，且参数顺序与初版 translateSubtitles(inputFile, outputFile) 一致
                input_srt_path = os.path.abspath(final_srt_path)
                output_srt_path = os.path.abspath(translated_srt_path)
                ok = translator.translateSubtitles(input_srt_path, output_srt_path)
                if not ok or not os.path.exists(translated_srt_path):
                    self.error_occurred.emit("翻译字幕失败（请检查有道 appId/secretKey 或网络）。")
                    return
                subtitle_file_name = os.path.basename(output_srt_path)
            else:
                subtitle_file_name = os.path.basename(final_srt_path)
            self.progress_percent.emit(85)

            # 5) 合成带字幕视频（FFmpeg 合成修复 + 字幕颜色修复）
            self.progress_updated.emit("正在合成带字幕的视频...")
            # 字幕颜色修复：ASS 格式为 &HAABBGGRR&（BGR），与 UI 选项一致
            color_map = {
                "白色": "&H00FFFFFF&",
                "黄色": "&H0000FFFF&",
                "青色": "&H00FFFF00&",
                "红色": "&H000000FF&",
                "绿色": "&H0000FF00&",
                "蓝色": "&H00FF0000&",
                "橙色": "&H0000A5FF&",
                "粉色": "&H00B469FF&",
                "紫色": "&H00800080&",
            }
            primary_colour = color_map.get(
                self.subtitle_color,
                color_map.get("白色", "&H00FFFFFF&"),
            )
            # 较快/极快模型单句往往更长，限制最大字号避免字幕超出画面
            effective_font_size = min(self.subtitle_font_size, 24) if self.whisper_model in ("tiny", "base") else self.subtitle_font_size
            force_style = (
                f"FontName={self.subtitle_font},"
                f"FontSize={effective_font_size},"
                f"PrimaryColour={primary_colour}"
            )
            safe_srt_name = subtitle_file_name.replace("'", r"\'")
            safe_style = force_style.replace("'", r"\'").replace("\\", "\\\\")
            # 将 force_style 拼接到 subtitles 滤镜；-preset veryfast 加快合成
            ffmpeg_cmd = (
                f'{ffmpeg} -y '
                f'-i "{self.video_path}" '
                f'-vf "subtitles=\'{safe_srt_name}\':force_style=\'{safe_style}\'" '
                f'-c:v libx264 -preset veryfast -threads 0 -c:a aac "{output_video_name}"'
            )
            print(f"[FFmpeg] 合成命令：{ffmpeg_cmd}")
            r = run_subprocess(ffmpeg_cmd, cwd=base_dir)
            if (not r.success) or (not os.path.exists(output_video_path)):
                self.error_occurred.emit(f"合成带字幕视频失败：\n{r.stderr}")
                return

            self.progress_percent.emit(100)
            self.finished.emit(output_video_path)

            # 冗余文件清理：仅保留原视频、带字幕视频、最终字幕；删除提取音频与中间字幕
            to_delete = [audio_path, raw_srt_path]
            if self.bilingual:
                to_delete.append(vad_srt_path)
            for path in to_delete:
                if not path or not os.path.exists(path):
                    continue
                try:
                    os.remove(path)
                    print(f"[视频字幕] 已清理临时文件：{path}")
                except Exception as e:
                    print(f"[视频字幕] 清理临时文件失败（已忽略）：{path}，{e}")
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"处理视频时发生异常：{e}")


# ======================
# 录音线程 & 窗口（未改主逻辑，仅复用 CUDA 检测）
# ======================

class RecordingThread(QThread):
    """录音线程：使用 PyAudio 流式读取，在读取循环内计算并发射音量，保证录制时进度条实时响应。"""
    update_audio_level = pyqtSignal(int)
    recording_stopped = pyqtSignal(object)
    status_updated = pyqtSignal(str)

    SAMPLE_RATE = 16000
    CHUNK_FRAMES = 1024
    SAMPLE_WIDTH = 2

    def __init__(self, device_index: int, lang: str, vad_mode: int):
        super().__init__()
        self.device_index = device_index
        self.lang = lang
        self.vad_mode = vad_mode
        self.is_recording = False
        self.stop_flag = False
        self.p = None
        self.stream = None

    def run(self):
        self.is_recording = True
        self.stop_flag = False
        chunks: List[bytes] = []
        try:
            self.status_updated.emit("正在调整环境噪音...")
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK_FRAMES,
            )
            # 短暂采集用于“环境噪音”提示，不用于音量条
            for _ in range(8):
                if self.stop_flag:
                    break
                self.stream.read(self.CHUNK_FRAMES, exception_on_overflow=False)
            self.status_updated.emit("开始录音，请说话...")

            while not self.stop_flag:
                try:
                    data = self.stream.read(self.CHUNK_FRAMES, exception_on_overflow=False)
                    chunks.append(data)
                    rms = _audioop_rms(data, self.SAMPLE_WIDTH)
                    level = min(100, int(rms / 32767 * 100))
                    self.update_audio_level.emit(level)
                except Exception as e:
                    if not self.stop_flag:
                        traceback.print_exc()
                    break

            if not chunks:
                self.recording_stopped.emit(None)
                return
            raw = b"".join(chunks)
            audio = sr.AudioData(raw, self.SAMPLE_RATE, self.SAMPLE_WIDTH)
            self.recording_stopped.emit(audio)
        except Exception as e:
            traceback.print_exc()
            self.status_updated.emit(f"录音错误: {e}")
            self.recording_stopped.emit(None)
        finally:
            self.is_recording = False
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            if self.p:
                try:
                    self.p.terminate()
                except Exception:
                    pass
                self.p = None

    def stop(self):
        self.stop_flag = True


class MicrophoneTestThread(QThread):
    update_audio_level = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    test_completed = pyqtSignal(bool)

    def __init__(self, device_index: int):
        super().__init__()
        self.device_index = device_index
        self.stop_flag = False
        self.max_level = 0
        self.p = None
        self.stream = None

    def run(self):
        try:
            self.status_updated.emit("正在测试麦克风...")
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=1024,
            )
            self.status_updated.emit("麦克风测试中，请说话...")

            start = time.time()
            while not self.stop_flag and (time.time() - start) < 10:
                data = self.stream.read(1024, exception_on_overflow=False)
                rms = _audioop_rms(data, 2)
                level = int(rms / 32767 * 100)
                self.max_level = max(self.max_level, level)
                self.update_audio_level.emit(level)
                time.sleep(0.1)

            self.test_completed.emit(self.max_level > 5)
        except Exception as e:
            traceback.print_exc()
            self.status_updated.emit(f"测试错误: {e}")
            self.test_completed.emit(False)
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()

    def stop(self):
        self.stop_flag = True


class RecordingWindow(QWidget):
    def __init__(self, whisper_model_getter):
        super().__init__()
        self.setWindowTitle("录音界面")
        self.resize(680, 520)
        self.get_whisper_model = whisper_model_getter

        layout = QVBoxLayout(self)

        lang_group = QGroupBox("识别语言")
        lang_group.setMinimumWidth(420)
        lang_layout = QHBoxLayout()
        self.lang_en = QRadioButton("英语")
        self.lang_zh = QRadioButton("中文")
        self.lang_auto = QRadioButton("自动检测")
        self.lang_en.setChecked(True)
        for w in (self.lang_en, self.lang_zh, self.lang_auto):
            w.setMinimumWidth(90)
            lang_layout.addWidget(w)
        lang_layout.addStretch(1)
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setMinimumHeight(22)
        self.progress_bar.setMinimumWidth(400)
        layout.addWidget(self.progress_bar)

        self.mic_status_label = QLabel("麦克风状态: 未连接")
        self.mic_status_label.setMinimumWidth(380)
        self.mic_status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.mic_status_label)

        self.microphone_combobox = QComboBox()
        self.microphone_combobox.setMinimumWidth(400)
        layout.addWidget(self.microphone_combobox)
        self._populate_microphones()

        vad_group = QGroupBox("VAD灵敏度设置")
        vad_group.setMinimumWidth(420)
        vad_layout = QVBoxLayout()
        self.vad_label = QLabel("灵敏度: 中等")
        self.vad_label.setMinimumWidth(120)
        self.vad_slider = QSlider(Qt.Horizontal)
        self.vad_slider.setRange(0, 3)
        self.vad_slider.setValue(1)
        self.vad_slider.valueChanged.connect(self._on_vad_changed)
        vad_layout.addWidget(self.vad_label)
        vad_layout.addWidget(self.vad_slider)
        vad_group.setLayout(vad_layout)
        layout.addWidget(vad_group)

        btn_layout = QHBoxLayout()
        self.test_mic_button = QPushButton("测试麦克风")
        self.start_button = QPushButton("开始录音")
        self.stop_button = QPushButton("停止录音")
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("保存录音到文件")

        style = "font-size: 22px; background-color: #4CAF50; color: white;"
        for b in (self.test_mic_button, self.start_button, self.stop_button, self.save_button):
            b.setStyleSheet(style)
            b.setMinimumWidth(140)
            b.setMinimumHeight(44)

        self.test_mic_button.clicked.connect(self.test_microphone)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.save_button.clicked.connect(self.save_recording)

        for b in (self.test_mic_button, self.start_button, self.stop_button, self.save_button):
            btn_layout.addWidget(b)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        result_group = QGroupBox("识别结果")
        result_group.setMinimumHeight(180)
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(140)
        self.result_text.setMinimumWidth(400)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group, 1)

        self.recording_thread: Optional[RecordingThread] = None
        self.test_thread: Optional[MicrophoneTestThread] = None
        self.vad_processor = VadProcessor(aggressiveness=1)
        self.last_filtered_wav: Optional[str] = None

    def _on_vad_changed(self, v: int):
        text = {0: "低", 1: "中等", 2: "高", 3: "极高"}.get(v, "中等")
        self.vad_label.setText(f"灵敏度: {text}")
        self.vad_processor = VadProcessor(aggressiveness=v)

    def _populate_microphones(self):
        try:
            p = pyaudio.PyAudio()
            info = p.get_host_api_info_by_index(0)
            count = info.get("deviceCount", 0)
            if count == 0:
                self.microphone_combobox.addItem("未检测到麦克风设备", -1)
                return
            default_device = p.get_default_input_device_info()
            self.microphone_combobox.addItem(f"默认麦克风: {default_device['name']}", default_device["index"])
            for i in range(count):
                d = p.get_device_info_by_host_api_device_index(0, i)
                if d.get("maxInputChannels", 0) > 0 and d["index"] != default_device["index"]:
                    self.microphone_combobox.addItem(d.get("name", f"设备{i}"), i)
            p.terminate()
        except Exception as e:
            self.microphone_combobox.addItem(f"获取麦克风列表失败: {e}", -1)

    def update_audio_level_ui(self, level: int):
        self.progress_bar.setValue(level)

    def update_status(self, text: str):
        self.mic_status_label.setText(f"麦克风状态: {text}")

    def start_recording(self):
        device_index = self.microphone_combobox.currentData()
        if device_index == -1:
            QMessageBox.warning(self, "警告", "请选择有效的麦克风设备")
            return

        if self.lang_en.isChecked():
            lang = "en"
        elif self.lang_zh.isChecked():
            lang = "zh"
        else:
            lang = "auto"

        self.stop_recording()
        self.recording_thread = RecordingThread(device_index, lang, self.vad_slider.value())
        self.recording_thread.update_audio_level.connect(self.update_audio_level_ui)
        self.recording_thread.status_updated.connect(self.update_status)
        self.recording_thread.recording_stopped.connect(self.on_recording_stopped)
        self.recording_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.test_mic_button.setEnabled(False)
        self.microphone_combobox.setEnabled(False)
        self.result_text.clear()

    def stop_recording(self):
        if self.recording_thread and self.recording_thread.is_recording:
            self.recording_thread.stop()
            self.update_status("正在处理录音...")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.test_mic_button.setEnabled(True)
        self.microphone_combobox.setEnabled(True)

    def on_recording_stopped(self, audio_data: Optional[sr.AudioData]):
        if audio_data is None:
            self.update_status("录音失败")
            return

        try:
            self.update_status("正在应用 VAD...")
            filtered = self.vad_processor.apply_vad_to_audio_data(audio_data)

            out_dir = os.getcwd()
            self.last_filtered_wav = os.path.join(out_dir, "recording_filtered.wav")
            with open(self.last_filtered_wav, "wb") as f:
                f.write(filtered.get_wav_data())

            # 「NumPy版本修复」：录音转文字时同样在调用 Whisper 前做 NumPy 版本校验
            ok_np, np_msg = check_numpy_for_whisper()
            print(f"[Whisper/NumPy] {np_msg}")
            if not ok_np:
                QMessageBox.critical(self, "错误", "Whisper 识别失败（NumPy 版本不兼容）：\n" + np_msg)
                self.update_status("Whisper 处理失败（NumPy 版本不兼容）")
                return

            model = self.get_whisper_model()
            gpu = detect_gpu_for_whisper(model)
            self.update_status(gpu.status_text)

            self.update_status("正在使用 Whisper 识别文本...")
            ok_fast = False
            if USE_FAST_WHISPER:
                ok_fast, _ = run_fast_whisper_api(
                    self.last_filtered_wav,
                    out_dir,
                    "txt",
                    model,
                    language=self._get_lang_for_whisper(),
                    device=gpu.device,
                )
            if not ok_fast:
                whisper_cmd_cuda = build_whisper_cmd(
                    self.last_filtered_wav,
                    out_dir,
                    "txt",
                    model,
                    language=self._get_lang_for_whisper(),
                    device="cuda",
                )
                r = run_subprocess(whisper_cmd_cuda)
                if not r.success:
                    whisper_cmd_cpu = build_whisper_cmd(
                        self.last_filtered_wav,
                        out_dir,
                        "txt",
                        model,
                        language=self._get_lang_for_whisper(),
                        device="cpu",
                    )
                    r2 = run_subprocess(whisper_cmd_cpu)
                    if not r2.success:
                        QMessageBox.critical(self, "错误", f"Whisper 识别失败：\n{r.stderr or r2.stderr}")
                        self.update_status("Whisper 处理失败")
                        return

            txt_path = os.path.join(out_dir, "recording_filtered.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.result_text.setText(f.read())
                self.update_status(f"识别完成: {txt_path}")
            else:
                self.update_status("Whisper 未生成 TXT 文件")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"处理录音时发生错误: {e}")
            self.update_status("处理失败")

    def _get_lang_for_whisper(self) -> Optional[str]:
        if self.lang_en.isChecked():
            return "en"
        if self.lang_zh.isChecked():
            return "zh"
        return None

    def save_recording(self):
        if not self.last_filtered_wav or not os.path.exists(self.last_filtered_wav):
            QMessageBox.information(self, "提示", "当前没有可保存的录音，请先完成一次录音。")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存录音文件", "recording_filtered.wav", "音频文件 (*.wav)")
        if not save_path:
            return
        try:
            with open(self.last_filtered_wav, "rb") as src, open(save_path, "wb") as dst:
                dst.write(src.read())
            QMessageBox.information(self, "成功", f"录音已保存到：\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存录音失败：{e}")

    def test_microphone(self):
        device_index = self.microphone_combobox.currentData()
        if device_index == -1:
            QMessageBox.warning(self, "警告", "请选择有效的麦克风设备")
            return
        self.stop_recording()
        self.test_thread = MicrophoneTestThread(device_index)
        self.test_thread.update_audio_level.connect(self.update_audio_level_ui)
        self.test_thread.status_updated.connect(self.update_status)
        self.test_thread.test_completed.connect(self.on_test_completed)
        QMessageBox.information(self, "麦克风测试", "请说话，观察音量条是否有反应。\n测试将在10秒后自动停止。")
        self.test_thread.start()
        self.test_mic_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.microphone_combobox.setEnabled(False)

    def on_test_completed(self, success: bool):
        self.test_mic_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.microphone_combobox.setEnabled(True)
        if success:
            self.update_status("测试成功")
            QMessageBox.information(self, "麦克风测试", "麦克风测试成功！")
        else:
            self.update_status("测试无响应")
            QMessageBox.warning(self, "麦克风测试", "麦克风测试未检测到声音，请检查设备与权限。")

    def closeEvent(self, event):
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()
            self.recording_thread.wait()
        if self.test_thread and self.test_thread.isRunning():
            self.test_thread.stop()
            self.test_thread.wait()
        event.accept()


# ======================
# 主窗口（显卡状态 + UI 入口）
# ======================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频字幕翻译工具")
        self.setFixedSize(960, 840)

        central = QWidget()
        central.setMinimumHeight(460)
        central.setMinimumWidth(640)
        central.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ---------- 功能类型选择（最顶部） ----------
        mode_group = QGroupBox("功能类型")
        mode_group.setMinimumWidth(480)
        mode_layout = QHBoxLayout()
        self.radio_subtitle_video = QRadioButton("生成字幕视频")
        self.radio_record = QRadioButton("实时录音转文字")
        self.radio_subtitle_video.setChecked(True)
        mode_layout.addWidget(self.radio_subtitle_video)
        mode_layout.addWidget(self.radio_record)
        mode_layout.addStretch(1)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # ---------- 「生成字幕视频」配置区（选中该功能时显示） ----------
        self.subtitle_video_panel = QWidget()
        self.subtitle_video_panel.setMinimumHeight(380)
        self.subtitle_video_panel.setMinimumWidth(520)
        self.subtitle_video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        subtitle_video_layout = QVBoxLayout(self.subtitle_video_panel)
        subtitle_video_layout.setSpacing(14)

        # 第一步：选择视频文件 + 显示完整路径
        file_group = QGroupBox("第一步：选择视频文件")
        file_group.setMinimumWidth(480)
        file_layout = QVBoxLayout()
        file_btn_row = QHBoxLayout()
        self.btn_select_video = QPushButton("选择视频文件")
        self.btn_select_video.setStyleSheet("")
        self.btn_select_video.clicked.connect(self._on_select_video_file)
        file_btn_row.addWidget(self.btn_select_video)
        file_btn_row.addStretch(1)
        file_layout.addLayout(file_btn_row)
        self.label_video_path = QLabel("未选择文件")
        self.label_video_path.setStyleSheet("color: #333; padding: 6px; background: #f5f5f5; border: 1px solid #ddd;")
        self.label_video_path.setWordWrap(True)
        self.label_video_path.setMinimumHeight(40)
        self.label_video_path.setMinimumWidth(400)
        file_layout.addWidget(self.label_video_path)
        file_group.setLayout(file_layout)
        subtitle_video_layout.addWidget(file_group)

        # 第二步：语言选择（仅保留 中文 / 英文 两个选项）
        lang_group = QGroupBox("第二步：字幕语言")
        lang_group.setMinimumWidth(480)
        lang_layout = QHBoxLayout()
        self.radio_lang_zh = QRadioButton("中文")
        self.radio_lang_en = QRadioButton("英文")
        self.radio_lang_zh.setChecked(True)
        self.radio_lang_zh.setMinimumWidth(80)
        self.radio_lang_en.setMinimumWidth(80)
        lang_layout.addWidget(self.radio_lang_zh)
        lang_layout.addWidget(self.radio_lang_en)
        lang_layout.addStretch(1)
        lang_group.setLayout(lang_layout)
        subtitle_video_layout.addWidget(lang_group)

        # 输出目录（合成视频保存位置，留空=原视频所在目录）
        out_dir_group = QGroupBox("输出目录（留空则保存到原视频所在目录）")
        out_dir_group.setMinimumWidth(480)
        out_dir_layout = QHBoxLayout()
        self.label_output_dir = QLabel("未选择（将保存到原视频目录）")
        self.label_output_dir.setStyleSheet("color: #333; padding: 4px; background: #f5f5f5; border: 1px solid #ddd;")
        self.label_output_dir.setWordWrap(True)
        self.label_output_dir.setMinimumHeight(28)
        self.label_output_dir.setMinimumWidth(280)
        self.btn_choose_output_dir = QPushButton("选择目录")
        self.btn_choose_output_dir.clicked.connect(self._on_choose_output_dir)
        out_dir_layout.addWidget(self.label_output_dir)
        out_dir_layout.addWidget(self.btn_choose_output_dir)
        out_dir_group.setLayout(out_dir_layout)
        subtitle_video_layout.addWidget(out_dir_group)

        # 第三步：字幕样式 + 生成速度
        style_group = QGroupBox("第三步：字幕样式与生成速度")
        style_group.setMinimumWidth(480)
        style_layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("字幕字号："))
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["14", "16", "18", "20", "22", "24", "26", "28", "30", "32", "36"])
        self.font_size_combo.setCurrentText("22")
        self.font_size_combo.setMinimumWidth(80)
        self.font_size_combo.setToolTip("较快/极快模式下建议选较小字号，避免字幕超出画面")
        row1.addWidget(self.font_size_combo)
        row1.addWidget(QLabel("字幕颜色："))
        self.font_color_combo = QComboBox()
        self.font_color_combo.addItems(["白色", "黄色", "青色", "红色", "绿色", "蓝色", "橙色", "粉色", "紫色"])
        self.font_color_combo.setMinimumWidth(100)
        row1.addWidget(self.font_color_combo)
        row1.addStretch(1)
        style_layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("生成速度："))
        self.model_combo = QComboBox()
        for disp, model in [("极速", "tiny"), ("快速", "base"), ("中等（默认）", "small"), ("慢速", "medium"), ("极慢", "large")]:
            self.model_combo.addItem(disp, model)
        self.model_combo.setCurrentIndex(2)
        self.model_combo.setMinimumWidth(100)
        self.model_combo.setToolTip("字幕准确度与生成速度有关：越慢越准，极速/快速适合预览")
        row2.addWidget(self.model_combo)
        speed_hint = QLabel("（准确度与速度有关，越慢越准）")
        speed_hint.setStyleSheet("color: #666;")
        row2.addWidget(speed_hint)
        row2.addStretch(1)
        style_layout.addLayout(row2)
        style_group.setLayout(style_layout)
        subtitle_video_layout.addWidget(style_group)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_start_generate = QPushButton("开始生成")
        self.btn_start_generate.setMinimumHeight(48)
        self.btn_start_generate.setFixedWidth(140)
        self.btn_start_generate.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_start_generate.clicked.connect(self._on_start_generate_clicked)
        btn_row.addWidget(self.btn_start_generate)
        self.btn_stop_generate = QPushButton("停止生成")
        self.btn_stop_generate.setMinimumHeight(48)
        self.btn_stop_generate.setFixedWidth(120)
        self.btn_stop_generate.setStyleSheet("background-color: #E53935; color: white;")
        self.btn_stop_generate.setEnabled(False)
        self.btn_stop_generate.clicked.connect(self._on_stop_generate_clicked)
        btn_row.addWidget(self.btn_stop_generate)
        btn_row.addStretch(1)
        subtitle_video_layout.addLayout(btn_row)
        subtitle_video_layout.addStretch(1)

        layout.addWidget(self.subtitle_video_panel, 1)

        # ---------- 录音面板（选中「实时录音转文字」时显示，不新开窗口） ----------
        self.record_panel = QWidget()
        self.record_panel.setMinimumHeight(420)
        self.record_panel.setMinimumWidth(520)
        self.record_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        record_layout = QVBoxLayout(self.record_panel)
        record_layout.setSpacing(12)

        record_lang_group = QGroupBox("识别语言")
        record_lang_group.setMinimumWidth(480)
        record_lang_layout = QHBoxLayout()
        self.record_lang_en = QRadioButton("英语")
        self.record_lang_zh = QRadioButton("中文")
        self.record_lang_auto = QRadioButton("自动检测")
        self.record_lang_en.setChecked(True)
        for w in (self.record_lang_en, self.record_lang_zh, self.record_lang_auto):
            w.setMinimumWidth(90)
            record_lang_layout.addWidget(w)
        record_lang_layout.addStretch(1)
        record_lang_group.setLayout(record_lang_layout)
        record_layout.addWidget(record_lang_group)

        record_speed_row = QHBoxLayout()
        record_speed_row.addWidget(QLabel("识别速度："))
        self.record_model_combo = QComboBox()
        for disp, model in [("极速", "tiny"), ("快速", "base"), ("中等（默认）", "small"), ("慢速", "medium"), ("极慢", "large")]:
            self.record_model_combo.addItem(disp, model)
        self.record_model_combo.setCurrentIndex(2)
        self.record_model_combo.setMinimumWidth(100)
        self.record_model_combo.setToolTip("准确度与速度有关，越慢越准")
        record_speed_row.addWidget(self.record_model_combo)
        record_speed_row.addStretch(1)
        record_layout.addLayout(record_speed_row)

        self.record_level_bar = QProgressBar()
        self.record_level_bar.setRange(0, 100)
        self.record_level_bar.setValue(0)
        self.record_level_bar.setMinimumHeight(22)
        self.record_level_bar.setMinimumWidth(400)
        record_layout.addWidget(self.record_level_bar)

        self.record_mic_status_label = QLabel("麦克风状态: 未连接")
        self.record_mic_status_label.setMinimumWidth(380)
        self.record_mic_status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        record_layout.addWidget(self.record_mic_status_label)
        self.record_mic_combo = QComboBox()
        self.record_mic_combo.setMinimumWidth(400)
        record_layout.addWidget(self.record_mic_combo)
        self._populate_record_microphones()

        record_vad_group = QGroupBox("VAD灵敏度设置")
        record_vad_group.setMinimumWidth(480)
        record_vad_layout = QVBoxLayout()
        self.record_vad_label = QLabel("灵敏度: 中等")
        self.record_vad_label.setMinimumWidth(120)
        self.record_vad_slider = QSlider(Qt.Horizontal)
        self.record_vad_slider.setRange(0, 3)
        self.record_vad_slider.setValue(1)
        self.record_vad_slider.valueChanged.connect(self._on_record_vad_changed)
        record_vad_layout.addWidget(self.record_vad_label)
        record_vad_layout.addWidget(self.record_vad_slider)
        record_vad_group.setLayout(record_vad_layout)
        record_layout.addWidget(record_vad_group)

        record_btn_layout = QHBoxLayout()
        self.record_test_mic_btn = QPushButton("测试麦克风")
        self.record_start_btn = QPushButton("开始录音")
        self.record_stop_btn = QPushButton("停止录音")
        self.record_stop_btn.setEnabled(False)
        self.record_save_btn = QPushButton("保存录音到文件")
        record_btn_style = "background-color: #4CAF50; color: white;"
        for b in (self.record_test_mic_btn, self.record_start_btn, self.record_stop_btn, self.record_save_btn):
            b.setStyleSheet(record_btn_style)
            b.setMinimumWidth(140)
            b.setMinimumHeight(44)
        self.record_test_mic_btn.clicked.connect(self.record_test_microphone)
        self.record_start_btn.clicked.connect(self.record_start_recording)
        self.record_stop_btn.clicked.connect(self.record_stop_recording)
        self.record_save_btn.clicked.connect(self.record_save_recording)
        for b in (self.record_test_mic_btn, self.record_start_btn, self.record_stop_btn, self.record_save_btn):
            record_btn_layout.addWidget(b)
        record_btn_layout.addStretch(1)
        record_layout.addLayout(record_btn_layout)

        record_result_group = QGroupBox("识别结果（可下拉滚动查看全文）")
        record_result_group.setMinimumHeight(200)
        record_result_layout = QVBoxLayout()
        self.record_result_text = QTextEdit()
        self.record_result_text.setReadOnly(True)
        self.record_result_text.setMinimumHeight(160)
        self.record_result_text.setMinimumWidth(400)
        self.record_result_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.record_result_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.record_result_text.setLineWrapMode(QTextEdit.WidgetWidth)
        record_result_layout.addWidget(self.record_result_text)
        record_result_group.setLayout(record_result_layout)
        record_layout.addWidget(record_result_group, 1)

        self.recording_thread: Optional[RecordingThread] = None
        self.record_test_thread: Optional[MicrophoneTestThread] = None
        self.record_vad_processor = VadProcessor(aggressiveness=1)
        self.record_last_filtered_wav: Optional[str] = None
        self._record_temp_files: List[str] = []
        self._record_saved = False

        layout.addWidget(self.record_panel, 1)

        # 当前显示的面板占满中间区域，随窗口缩放一起变大变小
        # 默认显示/隐藏：生成字幕视频显示配置区，录音显示录音按钮
        self.record_panel.setVisible(False)
        self.radio_subtitle_video.toggled.connect(self._on_mode_toggled)
        self.radio_record.toggled.connect(self._on_mode_toggled)

        # 已选视频路径（用于开始生成时传递）
        self._selected_video_path = ""
        self._selected_output_dir = ""

        # GPU 状态显示（放在配置区下方，两种模式都可见）
        self.gpu_status_label = QLabel("显卡状态：未检测")
        self.gpu_status_label.setStyleSheet("color: #333;")
        self.gpu_status_label.setMinimumWidth(400)
        self.gpu_status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.gpu_status_label)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("就绪")
        self.status_label.setMinimumWidth(200)
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.progress_bar, 0)

        self.app_id = "567fde2727e87c8b"
        self.secret_key = "mRNiQKw1unLhVjNu9joEhREuaiqBKlFK"

        self.worker: Optional[VideoSubtitleWorker] = None
        self._gpu_detail_log: str = ""
        self._shown_torch_cpu_tip = False  # 「CUDA修复」：避免 CPU 版提示反复弹出

        self.refresh_gpu_status()
        self.model_combo.currentIndexChanged.connect(lambda _: self.refresh_gpu_status())
        # 首次布局稳定：延迟强制对隐藏面板做一次布局计算，避免首次切回「视频字幕生成」时变形
        QTimer.singleShot(50, self._ensure_layout_initialized)

    def _on_mode_toggled(self):
        """功能类型切换：显示对应配置区；强制布局重算，避免首次切换变形"""
        if self.radio_subtitle_video.isChecked():
            # 从「实时录音转文字」切回时，停止录音/测试线程，避免后台继续占用麦克风
            try:
                self.record_stop_recording()
                if self.record_test_thread and self.record_test_thread.isRunning():
                    self.record_test_thread.stop()
                    self.record_test_thread.wait()
            except Exception:
                pass
            self.record_panel.setVisible(False)
            self.subtitle_video_panel.setVisible(True)
        else:
            self.subtitle_video_panel.setVisible(False)
            self.record_panel.setVisible(True)
        # 先隐藏再显示，避免时序导致布局错算；强制布局激活与重绘，消除首次切换变形
        central = self.centralWidget()
        if central and central.layout():
            central.layout().activate()
        central.updateGeometry()
        self.updateGeometry()
        central.update()
        QApplication.processEvents()

    def _ensure_layout_initialized(self):
        """首次显示后强制中央布局激活，使隐藏面板也完成尺寸计算，避免首次切回时变形。"""
        central = self.centralWidget()
        if central and central.layout():
            central.layout().activate()
            central.updateGeometry()
        self.updateGeometry()

    def _on_select_video_file(self):
        """选择视频文件并更新路径标签"""
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.mkv *.avi)")
        if path:
            self._selected_video_path = path
            self.label_video_path.setText(path)

    def _on_choose_output_dir(self):
        """选择合成视频的输出目录，留空则使用原视频目录"""
        start = self._selected_output_dir or (os.path.dirname(self._selected_video_path) if self._selected_video_path else "")
        path = QFileDialog.getExistingDirectory(self, "选择输出目录（带字幕视频将保存到此目录）", start)
        if path:
            self._selected_output_dir = path
            self.label_output_dir.setText(path)
            self.label_output_dir.setToolTip(path)

    def _on_start_generate_clicked(self):
        """开始生成：校验已选文件，按语言执行"""
        if not self._selected_video_path or not os.path.isfile(self._selected_video_path):
            QMessageBox.information(self, "提示", "请先选择视频文件。")
            return
        bilingual = self.radio_lang_zh.isChecked()
        output_dir = self._selected_output_dir.strip() if self._selected_output_dir else None
        self.start_video_job(bilingual, self._selected_video_path, output_dir)

    def _on_stop_generate_clicked(self):
        """随时停止当前字幕生成任务"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("正在停止…")
            self.btn_stop_generate.setEnabled(False)

    def refresh_gpu_status(self):
        model = self.model_combo.currentData() or self.model_combo.currentText()
        gpu = detect_gpu_for_whisper(model)
        self.gpu_status_label.setText(f"显卡状态：{gpu.status_text}")
        self._gpu_detail_log = gpu.detail_log

        # 「CUDA修复」：当检测到 torch 为 CPU 版时，弹出一次可复制的安装指令
        if (
            not gpu.ok
            and not self._shown_torch_cpu_tip
            and (gpu.torch_cuda_version is None)
            and (gpu.torch_version is not None)
        ):
            self._shown_torch_cpu_tip = True
            tip = (
                f"检测到当前 torch 版本为 CPU 版：{gpu.torch_version}\n\n"
                "如需使用 RTX 3060 Laptop GPU 进行 CUDA 加速，请安装带 CUDA 的 torch 版本，例如：\n\n"
                "pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 "
                "-f https://download.pytorch.org/whl/cu118/torch_stable.html\n\n"
                "同时确保已安装 CUDA Toolkit 11.8 及以上，并将 nvcc 加入 PATH。"
            )
            QMessageBox.information(self, "CUDA 环境检测", tip)

    def start_video_job(self, bilingual: bool, video_path: Optional[str] = None, output_dir: Optional[str] = None):
        """启动字幕生成任务。video_path 由调用方传入；output_dir 为空则使用原视频所在目录。"""
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.information(self, "提示", "请先选择有效视频文件。")
            return
        print(f"SSL Support: {QSslSocket.supportsSsl()}")

        font_size = int(self.font_size_combo.currentText())
        font_color = self.font_color_combo.currentText()
        model = self.model_combo.currentData() or self.model_combo.currentText()

        self.worker = VideoSubtitleWorker(
            video_path=video_path,
            output_dir=output_dir or "",
            bilingual=bilingual,
            app_id=self.app_id,
            secret_key=self.secret_key,
            vad_mode=1,
            whisper_model=model,
            subtitle_font="SimHei",
            subtitle_font_size=font_size,
            subtitle_color=font_color,
        )
        self.worker.progress_updated.connect(self.on_progress_text)
        self.worker.progress_percent.connect(self.progress_bar.setValue)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.cancelled.connect(self.on_generate_cancelled)
        self.worker.gpu_log_ready.connect(self.on_gpu_log)
        self.worker.gpu_status_ready.connect(self.on_gpu_status)

        self.btn_start_generate.setEnabled(False)
        self.btn_stop_generate.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("开始处理...")
        self.worker.start()

    def on_progress_text(self, text: str):
        self.status_label.setText(text)

    def on_gpu_log(self, log: str):
        self._gpu_detail_log = log

    def on_gpu_status(self, text: str):
        self.gpu_status_label.setText(f"显卡状态：{text}")

    def on_error(self, msg: str):
        self.btn_start_generate.setEnabled(True)
        self.btn_stop_generate.setEnabled(False)
        self.status_label.setText("失败")
        QMessageBox.critical(self, "错误", msg + ("\n\n[显卡检测日志]\n" + self._gpu_detail_log if self._gpu_detail_log else ""))
        self.progress_bar.setValue(0)

    def on_finished(self, output_path: str):
        self.btn_start_generate.setEnabled(True)
        self.btn_stop_generate.setEnabled(False)
        self.status_label.setText("完成")
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "成功", f"带字幕的视频生成成功：\n{output_path}")

    def on_generate_cancelled(self):
        self.btn_start_generate.setEnabled(True)
        self.btn_stop_generate.setEnabled(False)
        self.status_label.setText("已停止")
        self.progress_bar.setValue(0)

    def _on_record_vad_changed(self, v: int):
        text = {0: "低", 1: "中等", 2: "高", 3: "极高"}.get(v, "中等")
        self.record_vad_label.setText(f"灵敏度: {text}")
        self.record_vad_processor = VadProcessor(aggressiveness=v)

    def _populate_record_microphones(self):
        try:
            p = pyaudio.PyAudio()
            info = p.get_host_api_info_by_index(0)
            count = info.get("deviceCount", 0)
            if count == 0:
                self.record_mic_combo.addItem("未检测到麦克风设备", -1)
                return
            default_device = p.get_default_input_device_info()
            self.record_mic_combo.addItem(f"默认麦克风: {default_device['name']}", default_device["index"])
            for i in range(count):
                d = p.get_device_info_by_host_api_device_index(0, i)
                if d.get("maxInputChannels", 0) > 0 and d["index"] != default_device["index"]:
                    self.record_mic_combo.addItem(d.get("name", f"设备{i}"), i)
            p.terminate()
        except Exception as e:
            self.record_mic_combo.addItem(f"获取麦克风列表失败: {e}", -1)

    def record_update_audio_level_ui(self, level: int):
        self.record_level_bar.setValue(level)

    def record_update_status(self, text: str):
        self.record_mic_status_label.setText(f"麦克风状态: {text}")

    def record_get_lang_for_whisper(self) -> Optional[str]:
        if self.record_lang_en.isChecked():
            return "en"
        if self.record_lang_zh.isChecked():
            return "zh"
        return None

    def record_start_recording(self):
        device_index = self.record_mic_combo.currentData()
        if device_index == -1:
            QMessageBox.warning(self, "警告", "请选择有效的麦克风设备")
            return
        self.record_stop_recording()
        lang = "en" if self.record_lang_en.isChecked() else ("zh" if self.record_lang_zh.isChecked() else "auto")
        self.recording_thread = RecordingThread(device_index, lang, self.record_vad_slider.value())
        self.recording_thread.update_audio_level.connect(self.record_update_audio_level_ui)
        self.recording_thread.status_updated.connect(self.record_update_status)
        self.recording_thread.recording_stopped.connect(self.record_on_recording_stopped)
        self.recording_thread.start()
        self.record_start_btn.setEnabled(False)
        self.record_stop_btn.setEnabled(True)
        self.record_test_mic_btn.setEnabled(False)
        self.record_mic_combo.setEnabled(False)
        self.record_result_text.clear()
        self.record_level_bar.setValue(0)
        self._record_temp_files = []
        self._record_saved = False

    def record_stop_recording(self):
        if self.recording_thread and self.recording_thread.is_recording:
            self.recording_thread.stop()
            self.record_update_status("正在处理录音...")
        self.record_start_btn.setEnabled(True)
        self.record_stop_btn.setEnabled(False)
        self.record_test_mic_btn.setEnabled(True)
        self.record_mic_combo.setEnabled(True)

    def record_on_recording_stopped(self, audio_data: Optional[sr.AudioData]):
        if audio_data is None:
            self.record_update_status("录音失败")
            self.progress_bar.setValue(0)
            return
        try:
            self.status_label.setText("正在处理录音...")
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            self.record_update_status("正在应用 VAD...")
            filtered = self.record_vad_processor.apply_vad_to_audio_data(audio_data)
            out_dir = os.getcwd()
            self.record_last_filtered_wav = os.path.join(out_dir, "recording_filtered.wav")
            self._record_temp_files.append(self.record_last_filtered_wav)
            self._record_saved = False
            with open(self.record_last_filtered_wav, "wb") as f:
                f.write(filtered.get_wav_data())
            self.progress_bar.setValue(25)
            QApplication.processEvents()
            ok_np, np_msg = check_numpy_for_whisper()
            print(f"[Whisper/NumPy] {np_msg}")
            if not ok_np:
                QMessageBox.critical(self, "错误", "Whisper 识别失败（NumPy 版本不兼容）：\n" + np_msg)
                self.record_update_status("Whisper 处理失败（NumPy 版本不兼容）")
                self.progress_bar.setValue(0)
                return
            model = self.record_model_combo.currentData() or self.record_model_combo.currentText()
            gpu = detect_gpu_for_whisper(model)
            self.record_update_status(gpu.status_text)
            self.record_update_status("正在使用 Whisper 识别文本...")
            self.progress_bar.setValue(30)
            QApplication.processEvents()

            txt_path = os.path.join(out_dir, "recording_filtered.txt")

            def _record_progress_cb(msg: str, pct=None):
                self.status_label.setText(msg)
                if pct is not None:
                    self.progress_bar.setValue(min(85, pct))
                QApplication.processEvents()

            ok_fast = False
            if USE_FAST_WHISPER:
                ok_fast, _ = run_fast_whisper_api(
                    self.record_last_filtered_wav,
                    out_dir,
                    "txt",
                    model,
                    language=self.record_get_lang_for_whisper(),
                    device=gpu.device,
                )
            if not ok_fast:
                whisper_cmd_cuda = build_whisper_cmd(
                    self.record_last_filtered_wav,
                    out_dir,
                    "txt",
                    model,
                    language=self.record_get_lang_for_whisper(),
                    device="cuda",
                )
                r = run_subprocess_with_progress(
                    whisper_cmd_cuda,
                    progress_callback=_record_progress_cb,
                    interval_sec=10,
                    cwd=out_dir,
                )
                if not r.success:
                    whisper_cmd_cpu = build_whisper_cmd(
                        self.record_last_filtered_wav,
                        out_dir,
                        "txt",
                        model,
                        language=self.record_get_lang_for_whisper(),
                        device="cpu",
                    )
                    r2 = run_subprocess_with_progress(
                        whisper_cmd_cpu,
                        progress_callback=_record_progress_cb,
                        interval_sec=10,
                        cwd=out_dir,
                    )
                    if not r2.success:
                        QMessageBox.critical(self, "错误", f"Whisper 识别失败：\n{r.stderr or r2.stderr}")
                        self.record_update_status("Whisper 处理失败")
                        self.progress_bar.setValue(0)
                        return
            self.progress_bar.setValue(100)
            self.status_label.setText("就绪")
            if txt_path not in self._record_temp_files:
                self._record_temp_files.append(txt_path)
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.record_result_text.setText(f.read())
                self.record_update_status(f"识别完成: {txt_path}")
            else:
                self.record_update_status("Whisper 未生成 TXT 文件")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"处理录音时发生错误: {e}")
            self.record_update_status("处理失败")
            self.progress_bar.setValue(0)

    def record_save_recording(self):
        if not self.record_last_filtered_wav or not os.path.exists(self.record_last_filtered_wav):
            QMessageBox.information(self, "提示", "当前没有可保存的录音，请先完成一次录音。")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存录音文件", "recording_filtered.wav", "音频文件 (*.wav)")
        if not save_path:
            return
        try:
            with open(self.record_last_filtered_wav, "rb") as src, open(save_path, "wb") as dst:
                dst.write(src.read())
            self._record_saved = True
            QMessageBox.information(self, "成功", f"录音已保存到：\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存录音失败：{e}")

    def record_test_microphone(self):
        device_index = self.record_mic_combo.currentData()
        if device_index == -1:
            QMessageBox.warning(self, "警告", "请选择有效的麦克风设备")
            return
        self.record_stop_recording()
        self.record_test_thread = MicrophoneTestThread(device_index)
        self.record_test_thread.update_audio_level.connect(self.record_update_audio_level_ui)
        self.record_test_thread.status_updated.connect(self.record_update_status)
        self.record_test_thread.test_completed.connect(self.record_on_test_completed)
        QMessageBox.information(self, "麦克风测试", "请说话，观察音量条是否有反应。\n测试将在10秒后自动停止。")
        self.record_test_thread.start()
        self.record_test_mic_btn.setEnabled(False)
        self.record_start_btn.setEnabled(False)
        self.record_mic_combo.setEnabled(False)

    def record_on_test_completed(self, success: bool):
        self.record_test_mic_btn.setEnabled(True)
        self.record_start_btn.setEnabled(True)
        self.record_mic_combo.setEnabled(True)
        if success:
            self.record_update_status("测试成功")
            QMessageBox.information(self, "麦克风测试", "麦克风测试成功！")
        else:
            self.record_update_status("测试无响应")
            QMessageBox.warning(self, "麦克风测试", "麦克风测试未检测到声音，请检查设备与权限。")

    def closeEvent(self, event):
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()
            self.recording_thread.wait()
        if self.record_test_thread and self.record_test_thread.isRunning():
            self.record_test_thread.stop()
            self.record_test_thread.wait()
        if not self._record_saved and self._record_temp_files:
            for path in self._record_temp_files:
                if not path:
                    continue
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"[实时录制] 已清理未保存的临时文件：{path}")
                except Exception as e:
                    print(f"[实时录制] 清理临时文件失败（已忽略）：{path}，{e}")
        event.accept()


def main():
    app = QApplication(sys.argv)
    # 界面字体略放大，避免过小
    font = app.font()
    if font.pointSize() > 0:
        font.setPointSize(min(12, font.pointSize() + 2))
    else:
        font.setPixelSize(14)
    app.setFont(font)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


def _subprocess_fast_whisper_entry() -> int:
    """【0xC0000005 规避】子进程入口：仅执行 faster-whisper 转写并写文件，不启动 GUI。崩溃仅影响本进程。"""
    if "--subprocess-fast-whisper" not in sys.argv:
        return -1
    argv = sys.argv[sys.argv.index("--subprocess-fast-whisper") + 1:]
    if len(argv) < 6:
        print("Usage: ... --subprocess-fast-whisper <audio_path> <output_dir> <format> <model> <device> [language]", file=sys.stderr)
        return 1
    audio_path, output_dir, output_format, model, device = argv[0], argv[1], argv[2], argv[3], argv[4]
    language = argv[5] if len(argv) > 5 and argv[5] else None
    try:
        ok, err = _run_fast_whisper_api_impl(
            audio_path, output_dir, output_format, model, language, device, None
        )
        return 0 if ok else 1
    except Exception as e:
        traceback.print_exc()
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = _subprocess_fast_whisper_entry()
    if exit_code >= 0:
        sys.exit(exit_code)
    main()
