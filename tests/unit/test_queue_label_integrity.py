from __future__ import annotations

import importlib

import pytest
import torch

from drifting_models.data.queue import ClassConditionalSampleQueue, QueueConfig


class _StubProvider:
    def __init__(self, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self._batches = batches
        self._index = 0

    def next_batch(self, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = self._batches[self._index % len(self._batches)]
        self._index += 1
        return images.to(device), labels.to(device)


def _make_batch(labels: list[int], *, channels: int = 4, image_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    images = torch.zeros(len(labels), channels, image_size, image_size, dtype=torch.float32)
    for index, label in enumerate(labels):
        images[index].fill_(float(label))
    return images, labels_tensor


def _assert_class_samples_match_label(queue: ClassConditionalSampleQueue, *, label: int, samples: int = 3) -> None:
    class_labels = torch.tensor([label], dtype=torch.long)
    positives = queue.sample_positive_grouped(
        class_labels=class_labels,
        samples_per_group=samples,
        device=torch.device("cpu"),
    )[0]
    means = positives.mean(dim=(1, 2, 3))
    assert torch.allclose(means, torch.full_like(means, float(label)))


@pytest.mark.parametrize("module_name", ["scripts.train_latent", "scripts.train_pixel"])
def test_prime_queue_class_balanced_never_relabels(module_name: str) -> None:
    module = importlib.import_module(module_name)
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=6, per_class_capacity=16, global_capacity=64, store_device="cpu")
    )
    provider = _StubProvider(
        [
            _make_batch([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        ]
    )

    module._prime_queue(
        queue=queue,
        provider=provider,
        num_classes=6,
        sample_count=12,
        warmup_mode="class_balanced",
        warmup_min_per_class=2,
        device=torch.device("cpu"),
    )

    for label in range(6):
        assert queue.class_count(label) >= 2
        _assert_class_samples_match_label(queue, label=label, samples=2)


@pytest.mark.parametrize("module_name", ["scripts.train_latent", "scripts.train_pixel"])
def test_ensure_queue_has_labels_uses_true_labels(module_name: str) -> None:
    module = importlib.import_module(module_name)
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=16, global_capacity=64, store_device="cpu")
    )
    seed_images, seed_labels = _make_batch([0, 0, 0, 0])
    queue.push(seed_images, seed_labels)
    provider = _StubProvider([_make_batch([1, 2, 1, 2])])

    backfilled = module._ensure_queue_has_labels(
        queue=queue,
        class_labels=torch.tensor([0, 1, 2], dtype=torch.long),
        provider=provider,
        device=torch.device("cpu"),
    )

    assert backfilled == 2
    _assert_class_samples_match_label(queue, label=1, samples=2)
    _assert_class_samples_match_label(queue, label=2, samples=2)


@pytest.mark.parametrize("module_name", ["scripts.train_latent", "scripts.train_pixel"])
def test_ensure_queue_has_labels_respects_required_count(module_name: str) -> None:
    module = importlib.import_module(module_name)
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=16, global_capacity=64, store_device="cpu")
    )
    seed_images, seed_labels = _make_batch([0, 0, 0, 0])
    queue.push(seed_images, seed_labels)
    provider = _StubProvider([_make_batch([1, 2, 1, 2])])

    backfilled = module._ensure_queue_has_labels(
        queue=queue,
        class_labels=torch.tensor([1, 2], dtype=torch.long),
        provider=provider,
        required_count=2,
        device=torch.device("cpu"),
    )

    assert backfilled == 2
    assert queue.class_count(1) >= 2
    assert queue.class_count(2) >= 2
    _assert_class_samples_match_label(queue, label=1, samples=2)
    _assert_class_samples_match_label(queue, label=2, samples=2)


@pytest.mark.parametrize("module_name", ["scripts.train_latent", "scripts.train_pixel"])
def test_ensure_queue_has_labels_raises_when_missing_classes_unavailable(module_name: str) -> None:
    module = importlib.import_module(module_name)
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=16, global_capacity=64, store_device="cpu")
    )
    seed_images, seed_labels = _make_batch([0, 0, 0, 0])
    queue.push(seed_images, seed_labels)
    provider = _StubProvider([_make_batch([1, 1, 1, 1])])

    with pytest.raises(RuntimeError, match="unresolved labels"):
        module._ensure_queue_has_labels(
            queue=queue,
            class_labels=torch.tensor([0, 2], dtype=torch.long),
            provider=provider,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("module_name", ["scripts.train_latent", "scripts.train_pixel"])
def test_ensure_queue_has_labels_raises_when_required_count_unavailable(module_name: str) -> None:
    module = importlib.import_module(module_name)
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=16, global_capacity=64, store_device="cpu")
    )
    seed_images, seed_labels = _make_batch([0, 0, 0, 0])
    queue.push(seed_images, seed_labels)
    provider = _StubProvider([_make_batch([1, 1, 1, 1])])

    with pytest.raises(RuntimeError, match="required_count=2"):
        module._ensure_queue_has_labels(
            queue=queue,
            class_labels=torch.tensor([2], dtype=torch.long),
            provider=provider,
            required_count=2,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("module_name", ["scripts.train_latent", "scripts.train_pixel"])
def test_prime_queue_class_balanced_is_deterministic_given_provider_order(module_name: str) -> None:
    module = importlib.import_module(module_name)
    batches = [
        _make_batch([0, 0, 1, 1, 3, 3, 0, 1, 3, 0, 1, 3]),
        _make_batch([2, 2, 2, 2, 0, 1, 3, 2, 0, 1, 3, 2]),
    ]
    queue_a = ClassConditionalSampleQueue(
        QueueConfig(num_classes=4, per_class_capacity=32, global_capacity=64, store_device="cpu")
    )
    queue_b = ClassConditionalSampleQueue(
        QueueConfig(num_classes=4, per_class_capacity=32, global_capacity=64, store_device="cpu")
    )
    provider_a = _StubProvider(batches)
    provider_b = _StubProvider(batches)

    module._prime_queue(
        queue=queue_a,
        provider=provider_a,
        num_classes=4,
        sample_count=12,
        warmup_mode="class_balanced",
        warmup_min_per_class=2,
        device=torch.device("cpu"),
    )
    module._prime_queue(
        queue=queue_b,
        provider=provider_b,
        num_classes=4,
        sample_count=12,
        warmup_mode="class_balanced",
        warmup_min_per_class=2,
        device=torch.device("cpu"),
    )

    state_a = queue_a.state_dict()
    state_b = queue_b.state_dict()
    assert torch.equal(state_a["global_labels"], state_b["global_labels"])
    assert torch.equal(state_a["global_images"], state_b["global_images"])
