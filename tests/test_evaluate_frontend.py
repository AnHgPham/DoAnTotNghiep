from scripts.evaluate import build_encoder_for_eval, resolve_frontend_type


def test_resolve_dscnn_pcen_checkpoint_frontend():
    frontend_type, feature_type = resolve_frontend_type(
        {"model_family": "dscnn", "frontend_type": "mel_pcen", "feature_type": "mel"},
        {"model": {"architecture": "DSCNN-L"}},
        model_family="dscnn",
    )

    assert frontend_type == "mel_pcen"
    assert feature_type == "mel"


def test_build_dscnn_pcen_eval_encoder_shape_metadata():
    encoder = build_encoder_for_eval(
        model_family="dscnn",
        frontend_type="mel_pcen",
        cfg={"model": {"architecture": "DSCNN-L"}},
    )

    assert encoder.feature_type == "mel"
    assert encoder.frontend_type == "mel_pcen"
    assert encoder.use_pcen is True


def test_resolve_edgespot_legacy_mel_defaults_to_pcen():
    frontend_type, feature_type = resolve_frontend_type(
        {"model_family": "edgespot_full", "feature_type": "mel"},
        {"model": {"edge_use_pcen": True}},
        model_family="edgespot_full",
    )

    assert frontend_type == "mel_pcen"
    assert feature_type == "mel"
