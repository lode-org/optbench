"""The AKMC server import path follows the prefix that built eonclient."""

import os
import sys

from optbench.eon_akmc import next_search_batch, prepare_client_env


def test_prepare_client_env_reads_pixi_site_packages_and_pth(tmp_path):
    site = tmp_path / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    source = tmp_path / "schema" / "src"
    source.mkdir(parents=True)
    (site / "link.pth").write_text(f"{source}\n# comment\nimport not_a_path\n", encoding="utf-8")
    client = tmp_path / "bin" / "eonclient"
    client.parent.mkdir()
    client.write_bytes(b"")
    saved_path = list(sys.path)
    saved_env = {key: os.environ.get(key) for key in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH")}
    try:
        prepare_client_env(client)
        entries = os.environ["PYTHONPATH"].split(os.pathsep)
        assert entries[0] == str(source)
        assert str(site) in entries
        assert str(source) in sys.path
        assert str(site) in sys.path
    finally:
        sys.path[:] = saved_path
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_next_search_batch_stops_on_the_search_limit():
    assert next_search_batch(32, 0, 0) == 32
    assert next_search_batch(32, 0, 5000) == 32
    assert next_search_batch(32, 4976, 5000) == 24
    assert next_search_batch(32, 5000, 5000) == 0
