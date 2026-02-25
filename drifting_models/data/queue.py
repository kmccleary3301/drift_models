from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QueueConfig:
    num_classes: int
    per_class_capacity: int = 128
    global_capacity: int = 1000
    store_device: str = "cpu"
    strict_without_replacement: bool = False


class ClassConditionalSampleQueue:
    def __init__(self, config: QueueConfig) -> None:
        if config.num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if config.per_class_capacity <= 0:
            raise ValueError("per_class_capacity must be > 0")
        if config.global_capacity <= 0:
            raise ValueError("global_capacity must be > 0")
        self.config = config
        self._class_queues = [deque(maxlen=config.per_class_capacity) for _ in range(config.num_classes)]
        self._global_queue = deque(maxlen=config.global_capacity)
        self._global_labels = deque(maxlen=config.global_capacity)

    def push(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must be [B, C, H, W], got {tuple(images.shape)}")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError("labels must be [B] and aligned with images")
        stored_images = images.detach().to(self.config.store_device)
        for index in range(stored_images.shape[0]):
            label = int(labels[index].item())
            if label < 0 or label >= self.config.num_classes:
                raise ValueError(f"label out of range: {label}")
            sample = stored_images[index]
            self._class_queues[label].append(sample)
            self._global_queue.append(sample)
            self._global_labels.append(label)

    def sample_positive_grouped(
        self,
        class_labels: torch.Tensor,
        samples_per_group: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if class_labels.ndim != 1:
            raise ValueError("class_labels must be [G]")
        if samples_per_group <= 0:
            raise ValueError("samples_per_group must be > 0")
        groups = class_labels.shape[0]
        outputs = []
        for group_index in range(groups):
            label = int(class_labels[group_index].item())
            class_queue = self._class_queues[label]
            if len(class_queue) == 0:
                raise RuntimeError(f"class queue {label} is empty")
            sampled = _sample_from_queue(
                class_queue,
                samples_per_group,
                strict_without_replacement=bool(self.config.strict_without_replacement),
                queue_name=f"class queue {label}",
            )
            outputs.append(torch.stack(sampled, dim=0))
        return torch.stack(outputs, dim=0).to(device)

    def sample_unconditional_grouped(
        self,
        groups: int,
        samples_per_group: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if groups <= 0:
            raise ValueError("groups must be > 0")
        if samples_per_group <= 0:
            raise ValueError("samples_per_group must be > 0")
        if len(self._global_queue) == 0:
            raise RuntimeError("global queue is empty")
        outputs = []
        for _ in range(groups):
            sampled = _sample_from_queue(
                self._global_queue,
                samples_per_group,
                strict_without_replacement=bool(self.config.strict_without_replacement),
                queue_name="global queue",
            )
            outputs.append(torch.stack(sampled, dim=0))
        return torch.stack(outputs, dim=0).to(device)

    def class_count(self, label: int) -> int:
        if label < 0 or label >= self.config.num_classes:
            raise ValueError(f"label out of range: {label}")
        return len(self._class_queues[label])

    def global_count(self) -> int:
        return len(self._global_queue)

    def class_counts(self) -> list[int]:
        return [len(queue) for queue in self._class_queues]

    def state_dict(self) -> dict[str, Any]:
        global_images = None
        global_labels = None
        if len(self._global_queue) > 0:
            global_images = torch.stack(list(self._global_queue), dim=0).to("cpu")
            global_labels = torch.tensor(list(self._global_labels), dtype=torch.long)
        return {
            "version": 1,
            "config": {
                "num_classes": int(self.config.num_classes),
                "per_class_capacity": int(self.config.per_class_capacity),
                "global_capacity": int(self.config.global_capacity),
                "store_device": str(self.config.store_device),
                "strict_without_replacement": bool(self.config.strict_without_replacement),
            },
            "global_images": global_images,
            "global_labels": global_labels,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("queue state must be a dictionary")
        config_state = state.get("config")
        if not isinstance(config_state, dict):
            raise ValueError("queue state must include a config dictionary")
        expected = {
            "num_classes": int(self.config.num_classes),
            "per_class_capacity": int(self.config.per_class_capacity),
            "global_capacity": int(self.config.global_capacity),
            "store_device": str(self.config.store_device),
            "strict_without_replacement": bool(self.config.strict_without_replacement),
        }
        for key, expected_value in expected.items():
            actual_value = config_state.get(key)
            if key == "strict_without_replacement" and actual_value is None:
                actual_value = False
            if actual_value != expected_value:
                raise ValueError(
                    f"queue config mismatch for '{key}': expected {expected_value}, found {actual_value}"
                )
        global_images = state.get("global_images")
        global_labels = state.get("global_labels")
        for class_queue in self._class_queues:
            class_queue.clear()
        self._global_queue.clear()
        self._global_labels.clear()
        if global_images is None and global_labels is None:
            return
        if not isinstance(global_images, torch.Tensor) or not isinstance(global_labels, torch.Tensor):
            raise TypeError("queue state must include tensor global_images and global_labels")
        if global_labels.ndim != 1:
            raise ValueError("queue state global_labels must be 1D")
        if global_images.shape[0] != global_labels.shape[0]:
            raise ValueError("queue state global_images and global_labels length mismatch")
        if global_images.shape[0] == 0:
            return
        self.push(global_images, global_labels.to(dtype=torch.long, device=global_images.device))


@dataclass(frozen=True)
class GroupedSamplingConfig:
    positives_per_group: int
    unconditional_per_group: int


def sample_grouped_real_batches(
    *,
    queue: ClassConditionalSampleQueue,
    class_labels: torch.Tensor,
    config: GroupedSamplingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positives = queue.sample_positive_grouped(
        class_labels=class_labels,
        samples_per_group=config.positives_per_group,
        device=device,
    )
    unconditional = queue.sample_unconditional_grouped(
        groups=class_labels.shape[0],
        samples_per_group=config.unconditional_per_group,
        device=device,
    )
    return positives, unconditional


def _sample_from_queue(
    queue: deque[torch.Tensor],
    count: int,
    *,
    strict_without_replacement: bool = False,
    queue_name: str = "queue",
) -> list[torch.Tensor]:
    queue_len = len(queue)
    if queue_len == 0:
        raise RuntimeError("cannot sample from empty queue")
    if queue_len >= count:
        permutation = torch.randperm(queue_len)[:count]
        return [queue[index] for index in permutation.tolist()]
    if strict_without_replacement:
        raise RuntimeError(
            f"{queue_name} has {queue_len} samples but {count} requested with strict_without_replacement=true"
        )
    indices = torch.randint(0, queue_len, (count,))
    return [queue[index] for index in indices.tolist()]
