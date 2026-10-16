"""Public-package guard tests use artificial secret-like strings only."""
from pathlib import Path
from tools.check_publication import inspect, ROOT


def test_public_tree_contains_no_excluded_artifact_or_detected_secret():
    findings,_=inspect(ROOT)
    assert findings==[]


def test_forbidden_data_and_home_paths_are_flagged_without_echoing_values(tmp_path):
    (tmp_path/'data').mkdir()
    (tmp_path/'data'/'example.csv').write_text('synthetic')
    # Construct a fictional personal path so this test source is not a real leak.
    text='C:'+chr(92)+'Users'+chr(92)+'fictional-person'+chr(92)+'file'
    (tmp_path/'example.md').write_text(text)
    findings,_=inspect(tmp_path)
    assert {r['kind'] for r in findings}=={'excluded_artifact','personal_home_path'}
    assert text not in str(findings)


def test_synthetic_demo_checks_real_modules_without_market_data(capsys):
    from examples.synthetic_research_demo import main
    main()
    assert '"data": "synthetic_only"' in capsys.readouterr().out
