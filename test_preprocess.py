# -*- coding: utf-8 -*-
"""Быстрая проверка исправленного loudnorm-прохода (без ML, только ffmpeg).

Запуск из корня проекта:  python test_preprocess.py [путь_к.mp3]
Без аргумента — берётся data/lecture1.mp3.
"""
import sys
from pathlib import Path


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/lecture1.mp3")
    if not src.is_file():
        print(f"Файл не найден: {src}")
        return 2

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from audioanalytic.preprocess import preprocess
    from audioanalytic.winfix import find_ffmpeg

    ffmpeg = find_ffmpeg()
    dst = preprocess(src, ffmpeg=ffmpeg)
    print(f"OK: {dst.name} подготовлен (16 кГц, моно, ~-16 LUFS)")

    # Контроль результата через ffmpeg volumedetect
    import subprocess
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", str(dst),
                        "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    for line in p.stderr.splitlines():
        if "mean_volume" in line or "max_volume" in line:
            print("   ", line.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
