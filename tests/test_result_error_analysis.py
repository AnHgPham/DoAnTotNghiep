from scripts.analyze_result_errors import aggregate_confusion, aggregate_per_word


def test_result_error_analysis_aggregates_per_run_payloads():
    payload = {
        "per_run": [
            {
                "confusion": {"yes": {"yes": 8, "unknown": 2}, "cat": {"yes": 1, "unknown": 9}},
                "per_word": {
                    "yes": {"total": 10, "correct": 8, "rejected": 2, "confused": 0},
                    "cat": {"total": 10, "false_accept": 1, "correct_reject": 9},
                },
            },
            {
                "confusion": {"yes": {"yes": 9, "no": 1}, "cat": {"unknown": 10}},
                "per_word": {
                    "yes": {"total": 10, "correct": 9, "rejected": 0, "confused": 1},
                    "cat": {"total": 10, "false_accept": 0, "correct_reject": 10},
                },
            },
        ]
    }

    per_word = aggregate_per_word(payload)
    confusion = aggregate_confusion(payload)

    assert per_word["yes"]["total"] == 20
    assert per_word["yes"]["keyword_recall_at_far"] == 17 / 20
    assert per_word["cat"]["false_accept_rate"] == 1 / 20
    assert confusion[("yes", "unknown")] == 2
    assert confusion[("yes", "no")] == 1
