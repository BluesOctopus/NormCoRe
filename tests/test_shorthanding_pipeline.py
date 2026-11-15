from normcore_codec.shorthanding.pipeline import apply_shorthanding, restore_shorthanding


def test_shorthanding_pipeline_preserves_stage_order(monkeypatch):
    calls = []

    def fake_dynamic(text, registry, **kwargs):
        calls.append(("dynamic", text))
        return text + "D", {"dynamic": True}

    def fake_subtree(masked_source, text, **kwargs):
        calls.append(("subtree", text))
        return text + "S", {"subtree": True}

    monkeypatch.setattr("normcore_codec.shorthanding.pipeline.apply_registry_shorthanding", fake_dynamic)
    monkeypatch.setattr("normcore_codec.shorthanding.pipeline.apply_subtree_pass", fake_subtree)
    out, dynamic_stats, subtree_stats = apply_shorthanding(
        "source", "wire", [], count_fn=len, subtree_slots=8
    )
    assert out == "wireDS"
    assert calls == [("dynamic", "wire"), ("subtree", "wireD")]
    assert dynamic_stats == {"dynamic": True}
    assert subtree_stats == {"subtree": True}


def test_shorthanding_pipeline_inverts_in_reverse_order(monkeypatch):
    calls = []

    def fake_subtree(text):
        calls.append(("subtree", text))
        return text.removesuffix("S")

    def fake_dynamic(text, registry):
        calls.append(("dynamic", text))
        return text.removesuffix("D")

    monkeypatch.setattr("normcore_codec.shorthanding.pipeline.invert_subtree_pass", fake_subtree)
    monkeypatch.setattr("normcore_codec.shorthanding.pipeline.restore_registry_shorthanding", fake_dynamic)
    assert restore_shorthanding("wireDS", []) == "wire"
    assert calls == [("subtree", "wireDS"), ("dynamic", "wireD")]


def test_shorthanding_joint_best_of_selects_lowest_complete_wire(monkeypatch):
    def fake_dynamic(text, registry, **kwargs):
        if kwargs["scope_capture_slots"]:
            return "capture", {"strategy": "capture"}
        return "plain", {"strategy": "plain"}

    def fake_subtree(masked_source, text, **kwargs):
        return text + "+subtree", {"strategy": text + "+subtree"}

    costs = {
        "capture+subtree": 12,
        "plain+subtree": 7,
        "capture": 8,
        "plain": 10,
    }
    monkeypatch.setattr("normcore_codec.shorthanding.pipeline.apply_registry_shorthanding", fake_dynamic)
    monkeypatch.setattr("normcore_codec.shorthanding.pipeline.apply_subtree_pass", fake_subtree)
    out, dynamic_stats, subtree_stats = apply_shorthanding(
        "source",
        "wire",
        [],
        count_fn=costs.__getitem__,
        scope_capture_slots=32,
        subtree_slots=8,
        joint_best_of=True,
    )
    assert out == "plain+subtree"
    assert dynamic_stats["joint_selected_strategy"] == "patterns_subtree"
    assert dynamic_stats["joint_strategy_tokens"] == {
        "patterns_capture_subtree": 12,
        "patterns_subtree": 7,
        "patterns_capture": 8,
        "patterns": 10,
    }
    assert subtree_stats == {"strategy": "plain+subtree"}
