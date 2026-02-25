import torch
from PIL import Image

from drifting_models.data import (
    RealBatchProvider,
    RealBatchProviderConfig,
    SyntheticClassImageDataset,
    dataset_manifest_fingerprint,
)


def test_synthetic_dataset_shapes() -> None:
    dataset = SyntheticClassImageDataset(
        size=32,
        channels=3,
        image_size=16,
        num_classes=10,
        seed=7,
    )
    image, label = dataset[0]
    assert image.shape == (3, 16, 16)
    assert label.ndim == 0
    assert len(dataset) == 32


def test_real_batch_provider_returns_batches() -> None:
    provider = RealBatchProvider(
        RealBatchProviderConfig(
            source="synthetic_dataset",
            dataset_size=64,
            batch_size=8,
            seed=11,
            channels=4,
            image_size=8,
            num_classes=6,
        )
    )
    images, labels = provider.next_batch(device=torch.device("cpu"))
    assert images.shape == (8, 4, 8, 8)
    assert labels.shape == (8,)
    assert labels.min().item() >= 0
    assert labels.max().item() < 6


def test_imagefolder_provider_backend(tmp_path) -> None:
    class0 = tmp_path / "class0"
    class1 = tmp_path / "class1"
    class0.mkdir(parents=True, exist_ok=True)
    class1.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=(255, 0, 0)).save(class0 / "a.png")
    Image.new("RGB", (24, 24), color=(0, 255, 0)).save(class1 / "b.png")

    provider = RealBatchProvider(
        RealBatchProviderConfig(
            source="imagefolder",
            imagefolder_root=str(tmp_path),
            batch_size=2,
            channels=3,
            image_size=16,
            num_classes=2,
            transform_resize=20,
        )
    )
    images, labels = provider.next_batch(device=torch.device("cpu"))
    assert images.shape == (2, 3, 16, 16)
    assert labels.shape == (2,)
    assert int(labels.min().item()) >= 0


def test_tensor_file_provider_backend(tmp_path) -> None:
    tensor_path = tmp_path / "latent_dataset.pt"
    torch.save(
        {
            "images": torch.randn(10, 4, 8, 8),
            "labels": torch.randint(0, 5, (10,)),
        },
        tensor_path,
    )
    provider = RealBatchProvider(
        RealBatchProviderConfig(
            source="tensor_file",
            tensor_file_path=str(tensor_path),
            batch_size=4,
            channels=4,
            image_size=8,
            num_classes=5,
            shuffle=False,
        )
    )
    images, labels = provider.next_batch(device=torch.device("cpu"))
    assert images.shape == (4, 4, 8, 8)
    assert labels.shape == (4,)


def test_dataset_manifest_fingerprint_changes_with_data(tmp_path) -> None:
    class0 = tmp_path / "class0"
    class0.mkdir(parents=True, exist_ok=True)
    image_path = class0 / "x.png"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(image_path)

    config = RealBatchProviderConfig(
        source="imagefolder",
        imagefolder_root=str(tmp_path),
        batch_size=1,
        channels=3,
        image_size=16,
        num_classes=1,
    )
    fingerprint_a = dataset_manifest_fingerprint(config)
    Image.new("RGB", (16, 16), color=(9, 8, 7)).save(image_path)
    fingerprint_b = dataset_manifest_fingerprint(config)
    assert fingerprint_a != fingerprint_b


def test_webdataset_provider_requires_dependency() -> None:
    try:
        import webdataset  # noqa: F401
    except ModuleNotFoundError:
        provider_config = RealBatchProviderConfig(
            source="webdataset",
            webdataset_urls="dummy-{000000..000010}.tar",
            batch_size=2,
            channels=3,
            image_size=16,
            num_classes=10,
            dataset_size=32,
        )
        try:
            RealBatchProvider(provider_config)
        except RuntimeError as error:
            assert "webdataset package is required" in str(error)
        else:
            raise AssertionError("Expected RuntimeError when webdataset is unavailable")


def test_provider_determinism_with_fixed_seed() -> None:
    config = RealBatchProviderConfig(
        source="synthetic_dataset",
        dataset_size=128,
        batch_size=8,
        seed=123,
        channels=4,
        image_size=8,
        num_classes=10,
    )
    first_provider = RealBatchProvider(config)
    second_provider = RealBatchProvider(config)
    first_images, first_labels = first_provider.next_batch(device=torch.device("cpu"))
    second_images, second_labels = second_provider.next_batch(device=torch.device("cpu"))
    assert torch.allclose(first_images, second_images)
    assert torch.equal(first_labels, second_labels)


def test_prefetch_factor_requires_workers() -> None:
    config = RealBatchProviderConfig(
        source="synthetic_dataset",
        dataset_size=64,
        batch_size=8,
        num_workers=0,
        prefetch_factor=2,
    )
    try:
        RealBatchProvider(config)
    except ValueError as error:
        assert "prefetch_factor requires num_workers > 0" in str(error)
    else:
        raise AssertionError("Expected ValueError for prefetch_factor with num_workers=0")


def test_persistent_workers_requires_workers() -> None:
    config = RealBatchProviderConfig(
        source="synthetic_dataset",
        dataset_size=64,
        batch_size=8,
        num_workers=0,
        persistent_workers=True,
    )
    try:
        RealBatchProvider(config)
    except ValueError as error:
        assert "persistent_workers requires num_workers > 0" in str(error)
    else:
        raise AssertionError("Expected ValueError for persistent_workers with num_workers=0")
