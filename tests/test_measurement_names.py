"""Read old immutable snapshots through the current measurement vocabulary."""
import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from zoo_keeper import Zoo
from zoo.testing import SHA1, SHA2, SHA8


@pytest.mark.parametrize("version", ["0", "1", "2"])
def test_query_uses_current_columns_without_rewriting_snapshot(built_zoo, universe, tmp_path, version):
    path = tmp_path / "snapshot"
    shutil.copytree(built_zoo.resolve(), path)
    model_path = path / "models.parquet"
    table = pq.read_table(model_path)
    names = {"fingerprint_t1": "row_block_fingerprint",
             "fingerprint_t2": "full_weight_fingerprint", "spectra_t2": "full_weight_spectra"}
    frame = table.to_pandas().rename(columns=names)
    frame["coverage_status"] = frame["coverage_status"].replace({"t0_only": "metadata_only"})
    if version in {"0", "1"}:
        frame = frame.rename(columns={current: old for old, current in names.items()})
        frame["coverage_status"] = frame["coverage_status"].replace({"metadata_only": "t0_only"})
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), model_path)
    metadata_path = path / "zoo.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["zoo_version"] = version
    metadata_path.write_text(json.dumps(metadata))
    before = model_path.read_bytes(), metadata_path.read_bytes()
    with Zoo(path, data_root=universe.root) as opened:
        card = opened.info("org/child@" + SHA2)
        assert set(card["fingerprints"]) == {
            "row_block_fingerprint", "full_weight_fingerprint", "full_weight_spectra"}
        assert card["fingerprints"]["row_block_fingerprint"]
        assert opened.info("org/t0only@" + SHA8)["coverage"]["status"] == "metadata_only"
        assert len(opened.similar("org/child@" + SHA2)) > 0
        assert opened.diff("org/child@" + SHA2, "org/parent@" + SHA1)["summary"]["counts"]["changed"] > 0
    assert before == (model_path.read_bytes(), metadata_path.read_bytes())
