import json

from scripts.make_result_table import build_rows, markdown_table


def test_make_result_table_discovers_and_formats_metrics(tmp_path):
    result_dir = tmp_path / "results" / "edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_test100"
    result_dir.mkdir(parents=True)
    (result_dir / "gsc_edgespot_exact_k10_results.json").write_text(
        json.dumps(
            {
                "open_set_acc_at_1far": 0.8461333333333332,
                "open_set_acc_at_5far": 0.8611777777777777,
                "frr_at_5far": 0.21392727272727277,
                "auc": 0.9560512436363637,
                "eer": 0.11543854545454545,
                "keyword_acc": 0.7766181818181818,
                "f1": 0.8241123365438194,
                "per_run": [{}, {}],
            }
        ),
        encoding="utf-8",
    )

    rows = build_rows([], tmp_path / "results", manifest=None)
    table = markdown_table(rows)

    assert rows[0].label == "EdgeSpotFull T4 SCAF+GE2E"
    assert rows[0].split == "test"
    assert rows[0].runs == 100
    assert "86.12%" in table
    assert "77.66%" in table
    assert "82.41%" in table


def test_make_result_table_manifest_fallback(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "label": "DSCNN-L Triplet",
                        "split": "dev",
                        "runs": 30,
                        "open_set_acc_at_1far": 0.7647777777777777,
                        "open_set_acc_at_5far": 0.7916851851851849,
                        "frr_at_5far": 0.47909090909090907,
                        "auc": 0.8962673212121212,
                        "eer": 0.19767515151515153,
                        "keyword_acc": 0.6964848484848485,
                        "f1": 0.7127419277436541,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = build_rows([], tmp_path / "missing_results", manifest)

    assert len(rows) == 1
    assert rows[0].metrics["open_set_acc_at_5far"] == 0.7916851851851849
