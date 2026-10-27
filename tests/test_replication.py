"""Run the replication package's offline test suite under pytest as well."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_replication_offline_suite():
    sys.path.insert(0, str(ROOT / "replication" / "tests"))
    os.environ.setdefault("PDL_DATA", "")      # data-dependent cases skip unless a built data dir is given
    suite = unittest.defaultTestLoader.discover(str(ROOT / "replication" / "tests"), pattern="test_workflow.py")
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    assert result.wasSuccessful(), [str(f[0]) for f in result.failures + result.errors]
