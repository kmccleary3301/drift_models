import torch

from drifting_models.features import FeatureVectorizationConfig, vectorize_feature_maps


def test_vectorize_feature_maps_shapes() -> None:
    fmap0 = torch.randn(2, 8, 8, 8)
    fmap1 = torch.randn(2, 16, 4, 4)
    vectors = vectorize_feature_maps(
        [fmap0, fmap1],
        config=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=True,
            include_patch4_stats=True,
            include_input_x2_mean=True,
        ),
        input_images=torch.randn(2, 4, 16, 16),
    )

    assert "stage0.loc" in vectors
    assert "stage0.global" in vectors
    assert "stage0.patch2" in vectors
    assert "stage0.patch4" in vectors
    assert vectors["stage0.loc"].shape == (2, 64, 8)
    assert vectors["stage0.global"].shape == (2, 2, 8)
    assert vectors["stage1.loc"].shape == (2, 16, 16)
    assert vectors["input.x2mean"].shape == (2, 1, 4)


def test_vectorize_feature_maps_stage_selection() -> None:
    fmap0 = torch.randn(2, 8, 8, 8)
    fmap1 = torch.randn(2, 16, 4, 4)
    vectors = vectorize_feature_maps(
        [fmap0, fmap1],
        config=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=False,
            include_patch2_stats=False,
            include_patch4_stats=False,
            selected_stages=(1,),
        ),
    )
    assert "stage0.loc" not in vectors
    assert "stage1.loc" in vectors
