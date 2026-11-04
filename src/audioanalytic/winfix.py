"""Мелочи, без которых конвейер падает именно на Windows.

1) Консоль Windows (cp1251/cp866) может отказаться печатать юникод
   -> UnicodeEncodeError. Включаем UTF-8 для stdout/stderr.
2) WhisperX при диаризации вызывает torch.load, а чекпойнт pyannote хранит
   объекты вроде ``pyannote.core.segment.Segmentation``. Начиная с torch 2.6
   это приводит к UnpicklingError — разрешаем загрузку таких весов.
3) Проверка ffmpeg: понятная подсказка вместо непонятной ошибки внутри whisperx.
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
    """Разрешает torch.load грузить чекпойнты pyannote (WhisperX + torch>=2.6).

    Без этого диаризация падает с UnpicklingError: в весах pyannote лежат
    собственные объекты (Segmentation и т.п.), которые безопасная загрузка
    по умолчанию блокирует.
    """
    try:
        import torch
    except ImportError:
        return
    # pyannote должен быть импортирован до десериализации весов
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


# Типичные места, куда winget/сборщики кладут ffmpeg.exe (Windows)
_WIN_FFMPEG_GLOBS = [
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\**\bin\ffmpeg.exe",
    r"%ProgramFiles%\ffmpeg\**\bin\ffmpeg.exe",
    r"%ChocolateyInstall%\bin\ffmpeg.exe",
    r"%USERPROFILE%\scoop\apps\ffmpeg\current\bin\ffmpeg.exe",
]


def find_ffmpeg() -> str | None:
    """Полный путь к ffmpeg.exe: сначала PATH, потом типовые папки winget/scoop/choco.

    Возвращает путь к исполняемому файлу (а не к папке bin) — так его можно
    подставить напрямую в whisperx.load_audio(..., ffmpeg_file=...), даже если
    пользователь так и не смог добавить ffmpeg в PATH.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if os.name != "nt":
        return None
    for pattern in _WIN_FFMPEG_GLOBS:
        expanded = os.path.expandvars(pattern)
        try:
            matches = sorted(Path(expanded).parent.glob("**/" + Path(expanded).name))
        except OSError:
            continue
        if matches:
            return str(matches[0])
    return None


def ffmpeg_bin_dir() -> str | None:
    """Папка, содержащая ffmpeg.exe (нужна для подсказки по PATH)."""
    exe = find_ffmpeg()
    return str(Path(exe).parent) if exe else None


def path_add_hint(bin_dir: str) -> list[str]:
    """Команды, которые добавляют bin_dir в PATH пользователя без GUI."""
    return [
        f'setx PATH "%PATH%;{bin_dir}"   :: затем ЗАКРЫТЬ все окна cmd и открыть новое',
        f':: либо разово в текущем окне: set "PATH=%PATH%;{bin_dir}"',
    ]


def check_ffmpeg() -> None:
    """ffmpeg нужен для декодирования аудио — иначе whisperx.load_audio падает.

    Если ffmpeg не в PATH, но найден в типовых папках (winget и т.п.) —
    молча добавляем его папку в PATH текущего процесса: команды уже запущенного
    терминала (audioanalytic ...) будут работать без перезапуска окна.
    """
    if shutil.which("ffmpeg"):
        return
    exe = find_ffmpeg()
    if exe:
        bindir = str(Path(exe).parent)
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        print(f"[OK ] ffmpeg найден вне PATH, подключён автоматически: {exe}")
        print("      Чтобы работала команда `ffmpeg` в cmd — добавьте в PATH: " + bindir)
        return
    raise RuntimeError(
        "ffmpeg не найден (ни в PATH, ни в типовых папках winget/scoop/choco).\n"
        "Установите: winget install Gyan.FFmpeg, затем перезапустите терминал,\n"
        "либо скачайте сборку с ffmpeg.org и добавьте папку bin\\ в переменную PATH\n"
        "(sysdm.cpl → Дополнительно → Переменные среды → Path пользователя → Изменить → Создать)."
    )


# ── Место на диске ([Errno 28] No space left on device) ───────────────────────
# CUDA-сборка torch весит ~3.5 ГБ (в кэше pip + в site-packages нужно ~10-12 ГБ),
# веса моделей WhisperX/HF занимают ещё несколько ГБ. Всё это по умолчанию едет
# на диск C:, поэтому нехватка места — типичная проблема Windows.

MIN_FREE_GB_INSTALL = 20   # комфортный минимум для pip install с нуля
MIN_FREE_GB_RUN = 6        # минимум для скачивания весов моделей при прогоне


def free_gb(path: str | os.PathLike[str] | None = None) -> float | None:
    """Сколько ГБ свободно на диске, где лежит path (по умолчанию — текущая папка)."""
    try:
        usage = shutil.disk_usage(str(path or os.getcwd()))
    except OSError:
        return None
    return usage.free / 1024 ** 3


def cache_dirs() -> dict[str, str]:
    """Куда сейчас пишутся кэш pip и кэш Hugging Face (важно для диагностики)."""
    home = Path.home()
    if os.name == "nt":
        pip_cache = Path(os.environ.get("LOCALAPPDATA", home)) / "pip" / "Cache"
        hf_cache = Path(os.environ.get("HF_HOME", Path(os.environ.get("USERPROFILE", home)) / ".cache" / "huggingface"))
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
    """Команды переноса кэшей pip и HF на другой диск (например, D:\\).

    Достаточно задать одну переменную для каждого кэша; старые данные можно
    просто удалить — они скачаются заново уже в новое место.
    """
    return [
        'setx PIP_CACHE_DIR "D:\\pip-cache"   :: заменить D на нужный диск, перезапустить терминал',
        'setx HF_HOME "D:\\hf-cache"',
        'del /q "%LOCALAPPDATA%\\pip\\Cache\\*"* :: освободить старое место после переноса',
        'rmdir /s /q "%USERPROFILE%\\.cache\\huggingface"',
    ]


def disk_report_lines(min_free_run: float = MIN_FREE_GB_RUN) -> list[str]:
    """Строки для `audioanalytic doctor`: свободное место и размеры кэшей."""
    lines: list[str] = []
    free = free_gb()
    if free is not None:
        note = (f"{free:.1f} ГБ свободно"
                + ("" if free >= min_free_run else
                   f" — мало: для CUDA-torch нужно ~{MIN_FREE_GB_INSTALL} ГБ, см. README (раздел «Не хватает места»)"))
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
