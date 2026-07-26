from src.demo import api_server


def test_demo_exposes_only_two_featured_composite_profiles():
    featured = {
        profile_id: profile
        for profile_id, profile in api_server.MODEL_PROFILES.items()
        if profile.get("featured")
    }

    assert set(featured) == {
        "dscnn_pcen_ge2e",
        "edgespot_t4_pcen_ge2e",
    }
    assert featured["dscnn_pcen_ge2e"]["model_family"] == "dscnn"
    assert featured["edgespot_t4_pcen_ge2e"]["model_family"] == "edgespot_full"
    assert all(api_server.resolve_project_path(item["checkpoint"]).is_file()
               for item in featured.values())


def test_featured_profile_payload_contains_verified_metrics_and_flag():
    dscnn = api_server.model_profile_payload(
        "dscnn_pcen_ge2e",
        api_server.MODEL_PROFILES["dscnn_pcen_ge2e"],
    )
    edgespot = api_server.model_profile_payload(
        "edgespot_t4_pcen_ge2e",
        api_server.MODEL_PROFILES["edgespot_t4_pcen_ge2e"],
    )

    assert dscnn["featured"] is True
    assert edgespot["featured"] is True
    assert dscnn["metrics"][0] == {"label": "ACC@1%FAR", "value": "86.36%"}
    assert edgespot["metrics"][0] == {"label": "ACC@1%FAR", "value": "82.87%"}

