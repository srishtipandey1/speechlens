"""Fast unit tests for deterministic passage-level ladder generation."""

from collections import Counter

from scripts.make_dataset import FLAW_TYPES, _control_variants, generate_ladder, load_config
from speechlens.schema import WordTiming


def synthetic_timings(count: int = 180) -> list[WordTiming]:
    """Create widely spaced word timings for deterministic ladder tests."""
    return [
        WordTiming(
            word=f"word-{index}",
            start_s=index * 0.4,
            end_s=index * 0.4 + 0.25,
            confidence=0.9,
        )
        for index in range(count)
    ]


def test_ladder_is_seed_deterministic_balanced_and_non_overlapping() -> None:
    """Same seed returns identical flaws with balanced types and separated spans."""
    timings = synthetic_timings()
    joins = tuple(range(10, len(timings), 20))
    config = load_config(__import__("pathlib").Path.cwd())

    first = generate_ladder(timings, joins, config, seed=8128)
    second = generate_ladder(timings, joins, config, seed=8128)

    assert first == second
    all_specs = [spec for level in first for spec in level["flaws"]]
    counts = Counter(spec.flaw_type for spec in all_specs)
    assert set(counts) == set(FLAW_TYPES)
    assert max(counts.values()) - min(counts.values()) <= 1
    for level in first:
        specs = sorted(level["flaws"], key=lambda spec: spec.region)
        for earlier, later in zip(specs, specs[1:]):
            assert earlier.region[1] + config["ladder"]["minimum_region_separation_words"] <= later.region[0]
        assert all(
            spec.region[0] not in joins
            for spec in specs
            if spec.flaw_type in {"long_pause", "filler"}
        )


def test_control_variants_are_deterministic_for_the_same_seed() -> None:
    """All control audio and metadata repeat byte-identically for fixed inputs."""
    import numpy as np

    audio = synthetic_timings()
    waveform = np.sin(np.arange(16000, dtype=np.float32) / 37).astype(np.float32) * 0.2
    config = load_config(__import__("pathlib").Path.cwd())

    first = _control_variants(waveform, passage_index=2, seed=991, config=config)
    second = _control_variants(waveform, passage_index=2, seed=991, config=config)

    assert [item[0] for item in first] == [item[0] for item in second]
    assert [item[2] for item in first] == [item[2] for item in second]
    assert all(left[1].tobytes() == right[1].tobytes() for left, right in zip(first, second))