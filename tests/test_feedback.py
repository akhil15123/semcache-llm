import pytest

from semcache import SemanticCache


def test_wrong_hit_raises_threshold_and_blocks_repeat():
    cache = SemanticCache(threshold=0.5)
    cache.store("Turn on the lights", "lights on", namespace="home")
    hit = cache.lookup("Turn off the lights", namespace="home")
    assert hit is not None and hit.namespace == "home"

    new = cache.report_wrong_hit(hit)
    assert new == pytest.approx(hit.similarity + cache.feedback_margin)
    assert cache.lookup("Turn off the lights", namespace="home") is None
    assert cache.lookup("Turn on the lights", namespace="home").response == "lights on"
    assert cache.stats.wrong_hits == 1


def test_feedback_is_scoped_to_namespace():
    cache = SemanticCache(threshold=0.5)
    for ns in ("a", "b"):
        cache.store("Turn on the lights", "on", namespace=ns)
    cache.report_wrong_hit(cache.lookup("Turn off the lights", namespace="a"))
    assert cache.threshold_for("a") > 0.5
    assert cache.threshold_for("b") == 0.5
    assert cache.lookup("Turn off the lights", namespace="b") is not None


def test_threshold_never_lowered_by_a_weaker_wrong_hit():
    cache = SemanticCache(threshold=0.5)
    cache.store("Turn on the lights", "on")
    strong = cache.lookup("Turn on the lights")  # similarity ~1.0
    cache.report_wrong_hit(strong)
    assert cache.threshold_for() == 1.0
    cache._ns_threshold["default"] = 0.9
    from semcache import CacheHit

    cache.report_wrong_hit(CacheHit("x", 0.6, "p", "default"))
    assert cache.threshold_for() == 0.9


def test_confirmations_relax_back_to_base():
    cache = SemanticCache(threshold=0.8)
    cache.relax_step = 0.05
    cache._ns_threshold["default"] = 0.9
    from semcache import CacheHit

    hit = CacheHit("x", 0.95, "p", "default")
    assert cache.confirm_hit(hit) == pytest.approx(0.85)
    assert cache.confirm_hit(hit) == pytest.approx(0.8)
    assert cache.confirm_hit(hit) == 0.8
    assert "default" not in cache._ns_threshold
