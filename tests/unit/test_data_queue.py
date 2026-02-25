import pytest
import torch

from drifting_models.data import (
    ClassConditionalSampleQueue,
    GroupedSamplingConfig,
    QueueConfig,
    sample_grouped_real_batches,
)


def test_class_conditional_sample_queue_sampling_shapes() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=4, per_class_capacity=10, global_capacity=30, store_device="cpu")
    )
    images = torch.randn(20, 4, 8, 8)
    labels = torch.tensor([0, 1, 2, 3] * 5)
    queue.push(images, labels)

    class_labels = torch.tensor([0, 2, 3])
    positives = queue.sample_positive_grouped(class_labels, samples_per_group=3, device=torch.device("cpu"))
    unconditional = queue.sample_unconditional_grouped(groups=3, samples_per_group=2, device=torch.device("cpu"))

    assert positives.shape == (3, 3, 4, 8, 8)
    assert unconditional.shape == (3, 2, 4, 8, 8)
    assert queue.global_count() == 20
    assert queue.class_count(0) == 5


def test_sample_grouped_real_batches() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=20, store_device="cpu")
    )
    images = torch.randn(12, 4, 8, 8)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    queue.push(images, labels)

    class_labels = torch.tensor([0, 1])
    positives, unconditional = sample_grouped_real_batches(
        queue=queue,
        class_labels=class_labels,
        config=GroupedSamplingConfig(positives_per_group=2, unconditional_per_group=3),
        device=torch.device("cpu"),
    )
    assert positives.shape == (2, 2, 4, 8, 8)
    assert unconditional.shape == (2, 3, 4, 8, 8)


def test_queue_positive_sampling_is_class_consistent() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=24, store_device="cpu")
    )
    # Encode class identity into pixel values so we can verify selection without labels.
    images = torch.cat(
        [
            torch.zeros(6, 1, 4, 4) + 0.0,
            torch.zeros(6, 1, 4, 4) + 1.0,
            torch.zeros(6, 1, 4, 4) + 2.0,
        ],
        dim=0,
    )
    labels = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)
    queue.push(images, labels)

    class_labels = torch.tensor([0, 2])
    positives = queue.sample_positive_grouped(class_labels, samples_per_group=4, device=torch.device("cpu"))
    assert positives.shape == (2, 4, 1, 4, 4)

    mean0 = float(positives[0].mean().item())
    mean2 = float(positives[1].mean().item())
    assert abs(mean0 - 0.0) < 1e-6
    assert abs(mean2 - 2.0) < 1e-6


def test_queue_state_dict_roundtrip_preserves_sampling_behavior() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=20, store_device="cpu")
    )
    images = torch.arange(12 * 1 * 2 * 2, dtype=torch.float32).reshape(12, 1, 2, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
    queue.push(images, labels)

    state = queue.state_dict()
    restored = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=20, store_device="cpu")
    )
    restored.load_state_dict(state)

    assert restored.class_counts() == queue.class_counts()
    assert restored.global_count() == queue.global_count()

    class_labels = torch.tensor([0, 1, 2], dtype=torch.long)
    torch.manual_seed(123)
    positives_a = queue.sample_positive_grouped(class_labels, samples_per_group=2, device=torch.device("cpu"))
    torch.manual_seed(123)
    positives_b = restored.sample_positive_grouped(class_labels, samples_per_group=2, device=torch.device("cpu"))
    assert torch.allclose(positives_a, positives_b)

    torch.manual_seed(456)
    unconditional_a = queue.sample_unconditional_grouped(groups=2, samples_per_group=3, device=torch.device("cpu"))
    torch.manual_seed(456)
    unconditional_b = restored.sample_unconditional_grouped(groups=2, samples_per_group=3, device=torch.device("cpu"))
    assert torch.allclose(unconditional_a, unconditional_b)


def test_queue_state_dict_rejects_config_mismatch() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=20, store_device="cpu")
    )
    images = torch.randn(6, 1, 2, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    queue.push(images, labels)
    state = queue.state_dict()

    mismatched = ClassConditionalSampleQueue(
        QueueConfig(num_classes=4, per_class_capacity=8, global_capacity=20, store_device="cpu")
    )
    with pytest.raises(ValueError):
        mismatched.load_state_dict(state)


def test_queue_state_dict_backward_compat_without_strict_flag() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=20, store_device="cpu")
    )
    images = torch.randn(6, 1, 2, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    queue.push(images, labels)
    state = queue.state_dict()
    state["config"].pop("strict_without_replacement", None)

    restored = ClassConditionalSampleQueue(
        QueueConfig(num_classes=3, per_class_capacity=8, global_capacity=20, store_device="cpu")
    )
    restored.load_state_dict(state)
    assert restored.class_counts() == queue.class_counts()


def test_queue_strict_without_replacement_raises_on_class_underflow() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(
            num_classes=2,
            per_class_capacity=4,
            global_capacity=8,
            store_device="cpu",
            strict_without_replacement=True,
        )
    )
    images = torch.randn(3, 1, 4, 4)
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    queue.push(images, labels)

    with pytest.raises(RuntimeError, match="strict_without_replacement=true"):
        queue.sample_positive_grouped(
            torch.tensor([0], dtype=torch.long),
            samples_per_group=3,
            device=torch.device("cpu"),
        )


def test_queue_strict_without_replacement_raises_on_global_underflow() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(
            num_classes=2,
            per_class_capacity=4,
            global_capacity=8,
            store_device="cpu",
            strict_without_replacement=True,
        )
    )
    images = torch.randn(2, 1, 4, 4)
    labels = torch.tensor([0, 1], dtype=torch.long)
    queue.push(images, labels)

    with pytest.raises(RuntimeError, match="strict_without_replacement=true"):
        queue.sample_unconditional_grouped(
            groups=1,
            samples_per_group=3,
            device=torch.device("cpu"),
        )


def test_queue_strict_without_replacement_samples_without_duplicates_when_full() -> None:
    queue = ClassConditionalSampleQueue(
        QueueConfig(
            num_classes=1,
            per_class_capacity=4,
            global_capacity=8,
            store_device="cpu",
            strict_without_replacement=True,
        )
    )
    images = torch.arange(4 * 1 * 2 * 2, dtype=torch.float32).reshape(4, 1, 2, 2)
    labels = torch.zeros(4, dtype=torch.long)
    queue.push(images, labels)

    positives = queue.sample_positive_grouped(
        torch.tensor([0], dtype=torch.long),
        samples_per_group=4,
        device=torch.device("cpu"),
    )
    unique_rows = torch.unique(positives[0].reshape(4, -1), dim=0)
    assert unique_rows.shape[0] == 4
