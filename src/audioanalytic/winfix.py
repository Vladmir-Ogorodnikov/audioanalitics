"""Мелочи, без которых конвейер падает именно на Windows.

1) Консоль Windows (cp1251/cp866) может отказаться печатать юникод
   -> UnicodeEncodeError. Включаем UTF-8 для stdout/stderr.
2) WhisperX при диаризации вызывает torch.load, а чекпойнт pyannote хранит
   объекты вроде ``pyannote.core.segment.Segmentation``. Начиная с torch 2.6
   это приводит к UnpicklingError — разрешаем загрузку таких весов.
3) Проверка ffmpeg: понятная подсказка вместо непонятной ошибки внутри whisperx,
   плюс автоматический fallback на статический бинарник imageio-ffmpeg (если установлен).
4) Нехватка места на диске ([Errno 28]): проверка свободных ГБ + перенос кэшей pip и
   Hugging Face на другой диск одной переменной окружения (см. cache_env_lines).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def setup_console() -> None:
    """UTF-8 в консоли Windows (Python 3.7+). Безопасно и на Linux/macOS."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # уже переопределено / не tty
                pass


def allow_pyannote_unpickling() -> None:
    """Разрешает torch.load грузить чекпойнты pyannote (WhisperX + torch>=2.6)."""
    try:
        import torch
    except ImportError:
        return
    try:
        import pyannote.core  # noqa: F401
        import pyannote.audio  # noqa: F401
    except Exception:
        pass
    if getattr(torch.load, "_audioanalytic_patched", False):
        return
    orig = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    patched._audioanalytic_patched = True  # type: ignore[attr-defined]
    torch.load = patched  # type: ignore[assignment]


_WIN_FFMPEG_GLOBS = [
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\**\bin\ffmpeg.exe",
    r"%ProgramFiles%\ffmpeg\**\bin\ffmpeg.exe",
    r"%ChocolateyInstall%\bin\ffmpeg.exe",
    r"%USERPROFILE%\scoop\apps\ffmpeg\current\bin\ffmpeg.exe",
]


def find_ffmpeg() -> str | None:
    """Полный путь к ffmpeg: сначала PATH, затем winget/scoop/choco и imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if os.name == "nt":
        for pattern in _WIN_FFMPEG_GLOBS:
            expanded = os.path.expandvars(pattern)
            try:
                matches = sorted(Path(expanded).parent.glob("**/" + Path(expanded).name))
            except OSError:
                continue
            if matches:
                return str(matches[0])
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return str(bundled)
    except Exception:
        pass
    return None


def ffmpeg_bin_dir() -> str | None:
    """Папка, содержащая ffmpeg (нужна для подсказки по PATH)."""
    exe = find_ffmpeg()
    return str(Path(exe).parent) if exe else None


def path_add_hint(bin_dir: str) -> list[str]:
    """Команды, которые добавляют bin_dir в PATH пользователя без GUI."""
    return [
        f'setx PATH "%PATH%;{bin_dir}"   :: затем ЗАКРЫТЬ все окна cmd и открыть новое',
        f':: либо разово в текущем окне: set "PATH=%PATH%;{bin_dir}"',
    ]


def check_ffmpeg() -> None:
    """ffmpeg нужен для декодирования и нормализации аудио (-16 LUFS)."""
    if shutil.which("ffmpeg"):
        return
    exe = find_ffmpeg()
    if exe:
        bindir = str(Path(exe).parent)
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        print(f"[OK ] ffmpeg найден вне PATH, подключён автоматически: {exe}")
        return
    raise RuntimeError(
        "ffmpeg не найден (ни в PATH, ни в типовых папках winget/scoop/choco, ни в imageio-ffmpeg).\n"
        "Установите: winget install Gyan.FFmpeg, затем перезапустите терминал,\n"
        "либо установите пакет pip install imageio-ffmpeg."
    )


MIN_FREE_GB_INSTALL = 20
MIN_FREE_GB_RUN = 6


def free_gb(path: str | os.PathLike[str] | None = None) -> float | None:
    """Сколько ГБ свободно на диске, где лежит path (по умолчанию — текущая папка)."""
    try:
        usage = shutil.disk_usage(str(path or os.getcwd()))
    except OSError:
        return None
    return usage.free / 1024 ** 3


def cache_dirs() -> dict[str, str]:
    """Куда сейчас пишутся кэш pip и кэш Hugging Face."""
    home = Path.home()
    if os.name == "nt":
        pip_cache = Path(os.environ.get("LOCALAPPDATA", home)) / "pip" / "Cache"
        hf_cache = Path(
            os.environ.get("HF_HOME", Path(os.environ.get("USERPROFILE", home)) / ".cache" / "huggingface")
        )
    else:
        pip_cache = Path(os.environ.get("PIP_CACHE_DIR", home / ".cache" / "pip"))
        hf_cache = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface"))
    return {"pip": str(pip_cache), "huggingface": str(hf_cache)}


def dir_size_gb(path: str | os.PathLike[str]) -> float:
    """Приблизительный размер дерева каталогов в ГБ (несуществующая папка = 0)."""
    total = 0
    p = Path(path)
    if not p.exists():
        return 0.0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024 ** 3


def cache_env_lines(target_dir: str) -> list[str]:
    """Команды переноса кэшей pip и HF на другой диск (например, D:\\)."""
    return [
        'setx PIP_CACHE_DIR "D:\\pip-cache"   :: заменить D на нужный диск, перезапустить терминал',
        'setx HF_HOME "D:\\hf-cache"',
        'del /q "%LOCALAPPDATA%\\pip\\Cache\\*"* :: освободить старое место после переноса',
        'rmdir /s /q "%USERPROFILE%\\.cache\\huggingface"',
    ]


def disk_report_lines(min_free_run: float = MIN_FREE_GB_RUN) -> list[tuple[str, bool, str]]:
    """Строки для `audioanalytic doctor`: свободное место и размеры кэшей."""
    lines: list[tuple[str, bool, str]] = []
    free = free_gb()
    if free is not None:
        note = (
            f"{free:.1f} ГБ свободно"
            + (
                ""
                if free >= min_free_run
                else f" — мало: для CUDA-torch нужно ~{MIN_FREE_GB_INSTALL} ГБ, см. README"
            )
        )
        lines.append(("Диск (текущая папка)", free >= min_free_run, note))
    for name, d in cache_dirs().items():
        size = dir_size_gb(d)
        lines.append((f"Кэш {name}", True, f"{d} — {size:.1f} ГБ занято"))
    return lines


def ensure_disk_space(min_free_gb: float, what: str) -> None:
    """Понятная ошибка вместо Errno 28 в середине скачивания многостраничных весов."""
    free = free_gb()
    if free is not None and free < min_free_gb:
        raise RuntimeError(
            f"На диске осталось {free:.1f} ГБ, а для «{what}» нужно минимум {min_free_gb:.0f} ГБ.\n"
            "Освободите место или перенесите кэши на другой диск:\n"
            + "\n".join("  " + s for s in cache_env_lines(""))
        )
