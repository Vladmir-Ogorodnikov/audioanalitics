"""Кэш загруженных ИИ-моделей в памяти процесса с контролем VRAM и автоочисткой при простое.

На коротких записях (1–5 мин) чтение весов Whisper + wav2vec2 + pyannote с диска занимает
больше времени, чем само распознавание (45–90 с против 5–15 с вычислений).
Кэш удерживает модели между задачами очереди и автоматически выгружает их:
  * после N секунд простоя очереди (по умолчанию 300 с), чтобы не занимать память ПК;
  * при профиле ``low_vram`` (GPU ≤ 8 ГБ), где модели не должны лежать в VRAM одновременно;
  * при перехвате ошибки нехватки памяти (CUDA OOM).
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


def free_torch_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def has_ample_vram(min_gb: float = 10.0) -> bool:
    """Возвращает True для CPU или для CUDA GPU с объёмом памяти не менее ``min_gb`` ГБ."""
    try:
        import torch
        if not torch.cuda.is_available():
            return True
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return total_gb >= min_gb
    except Exception:
        return True


class ModelCache:
    """Потокобезопасный реестр разогретых моделей для воркера очереди."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[tuple[Any, ...], Any] = {}
        self._last_used: float = 0.0

    def get_or_load(
        self,
        key: tuple[Any, ...],
        loader: Callable[[], Any],
        *,
        exclusive: bool = False,
    ) -> Any:
        """Возвращает модель из кэша или загружает её через ``loader()``.

        Если ``exclusive=True`` (например, на GPU 8 ГБ), перед загрузкой новой модели
        все остальные модели выгружаются из памяти.
        """
        with self._lock:
            self._last_used = time.monotonic()
            if key in self._items:
                return self._items[key]
            if exclusive and self._items:
                self._items.clear()
                free_torch_memory()

        try:
            instance = loader()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                log.warning("Нехватка памяти при загрузке %s — очищаю кэш моделей и повторяю", key)
                self.clear()
                instance = loader()
            else:
                raise

        with self._lock:
            self._items[key] = instance
            self._last_used = time.monotonic()
        return instance

    def clear(self) -> int:
        """Полностью выгружает все кэшированные модели и освобождает VRAM/RAM."""
        with self._lock:
            count = len(self._items)
            self._items.clear()
        if count:
            free_torch_memory()
        return count

    def evict_if_idle(self, max_idle_sec: float = 300.0) -> int:
        """Выгружает модели, если кэш не использовался дольше ``max_idle_sec`` секунд."""
        with self._lock:
            if not self._items:
                return 0
            idle = time.monotonic() - self._last_used
            if idle < max_idle_sec:
                return 0
            count = len(self._items)
            self._items.clear()
        free_torch_memory()
        log.info("Кэш моделей очищен после %.0f с простоя (%d объектов)", idle, count)
        return count

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)


GLOBAL_MODEL_CACHE = ModelCache()
