from normcore_codec.passb import apply_pass_b, invert_pass_b


def test_pass_b_facade_preserves_stage_order(monkeypatch):
    calls = []

    def fake_dynamic(text, registry, **kwargs):
        calls.append(("dynamic", text))
        return text + "D", {"dynamic": True}

    def fake_subtree(masked_source, text, **kwargs):
        calls.append(("subtree", text))
        return text + "S", {"subtree": True}

    monkeypatch.setattr("normcore_codec.passb.apply_dynamic_pass", fake_dynamic)
    monkeypatch.setattr("normcore_codec.passb.apply_subtree_pass", fake_subtree)
    out, dynamic_stats, subtree_stats = apply_pass_b(
        "source", "wire", [], count_fn=len, subtree_slots=8
    )
    assert out == "wireDS"
    assert calls == [("dynamic", "wire"), ("subtree", "wireD")]
    assert dynamic_stats == {"dynamic": True}
    assert subtree_stats == {"subtree": True}


def test_pass_b_facade_inverts_in_reverse_order(monkeypatch):
    calls = []

    def fake_subtree(text):
        calls.append(("subtree", text))
        return text.removesuffix("S")

    def fake_dynamic(text, registry):
        calls.append(("dynamic", text))
        return text.removesuffix("D")

    monkeypatch.setattr("normcore_codec.passb.invert_subtree_pass", fake_subtree)
    monkeypatch.setattr("normcore_codec.passb.invert_dynamic_pass", fake_dynamic)
    assert invert_pass_b("wireDS", []) == "wire"
    assert calls == [("subtree", "wireDS"), ("dynamic", "wireD")]


def test_pass_b_joint_best_of_selects_lowest_complete_wire(monkeypatch):
    def fake_dynamic(text, registry, **kwargs):
        if kwargs["scope_capture_slots"]:
            return "capture", {"arm": "capture"}
        return "plain", {"arm": "plain"}

    def fake_subtree(masked_source, text, **kwargs):
        return text + "+subtree", {"arm": text + "+subtree"}

    costs = {
        "capture+subtree": 12,
        "plain+subtree": 7,
        "capture": 8,
        "plain": 10,
    }
    monkeypatch.setattr("normcore_codec.passb.apply_dynamic_pass", fake_dynamic)
    monkeypatch.setattr("normcore_codec.passb.apply_subtree_pass", fake_subtree)
    out, dynamic_stats, subtree_stats = apply_pass_b(
        "source",
        "wire",
        [],
        count_fn=costs.__getitem__,
        scope_capture_slots=32,
        subtree_slots=8,
        joint_best_of=True,
    )
    assert out == "plain+subtree"
    assert dynamic_stats["joint_selected_arm"] == "patterns_subtree"
    assert dynamic_stats["joint_arm_tokens"] == {
        "patterns_capture_subtree": 12,
        "patterns_subtree": 7,
        "patterns_capture": 8,
        "patterns": 10,
    }
    assert subtree_stats == {"arm": "plain+subtree"}
