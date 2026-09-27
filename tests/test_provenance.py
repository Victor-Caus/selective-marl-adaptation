import json
import re
from pathlib import Path


def test_reference_manifest_has_immutable_provenance() -> None:
    manifest = json.loads(Path("reproductions/references.lock.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert len(manifest["sources"]) >= 4
    for source in manifest["sources"]:
        assert source["repository"].startswith("https://github.com/")
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
        assert source["license"]
        assert source["citation_key"]
        assert source["role"]
