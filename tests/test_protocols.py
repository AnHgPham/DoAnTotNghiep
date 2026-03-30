"""Tests for evaluation protocols."""

from src.evaluation.protocols import (
    EvaluationProtocol,
    GSC_ALL_35_WORDS,
    GSC_EXCLUDED_WORDS,
    GSC_POSITIVE_WORDS,
)


def test_gsc_fixed_partition():
    proto = EvaluationProtocol(dataset="gsc", mode="fixed")
    pos, neg = proto.get_partitions(0)
    assert len(pos) == 10
    assert len(neg) == 20
    assert set(pos) == set(GSC_POSITIVE_WORDS)
    for w in neg:
        assert w not in GSC_EXCLUDED_WORDS
        assert w not in pos


def test_gsc_fixed_deterministic():
    proto = EvaluationProtocol(dataset="gsc", mode="fixed")
    pos1, neg1 = proto.get_partitions(0)
    pos2, neg2 = proto.get_partitions(1)
    assert pos1 == pos2
    assert neg1 == neg2


def test_gsc_random_partition():
    proto = EvaluationProtocol(dataset="gsc", mode="random")
    pos, neg = proto.get_partitions(0)
    assert len(pos) == 10
    assert len(neg) == 20
    for w in pos + neg:
        assert w not in GSC_EXCLUDED_WORDS


def test_gsc_random_different_runs():
    proto = EvaluationProtocol(dataset="gsc", mode="random")
    pos0, neg0 = proto.get_partitions(0)
    pos1, neg1 = proto.get_partitions(1)
    assert (pos0, neg0) != (pos1, neg1)


def test_gsc_random_reproducible():
    proto1 = EvaluationProtocol(dataset="gsc", mode="random", seed=42)
    proto2 = EvaluationProtocol(dataset="gsc", mode="random", seed=42)
    pos1, neg1 = proto1.get_partitions(0)
    pos2, neg2 = proto2.get_partitions(0)
    assert pos1 == pos2
    assert neg1 == neg2


def test_gsc_no_overlap():
    proto = EvaluationProtocol(dataset="gsc", mode="random")
    for run_idx in range(5):
        pos, neg = proto.get_partitions(run_idx)
        assert len(set(pos) & set(neg)) == 0


def test_gsc_all_35_words():
    assert len(GSC_ALL_35_WORDS) == 35
    assert len(GSC_POSITIVE_WORDS) == 10
    assert len(GSC_EXCLUDED_WORDS) == 5


def test_protocol_n_runs():
    proto = EvaluationProtocol(dataset="gsc", mode="fixed", n_runs=10)
    assert proto.n_runs == 10


def test_invalid_dataset():
    try:
        EvaluationProtocol(dataset="invalid", mode="fixed")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
