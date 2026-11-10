"""Checkpoint transfers support both the driver and historical cluster layout."""
from types import SimpleNamespace

import pytest

import olivia


@pytest.mark.parametrize("layout", ["", "/capture"])
def test_remote_checkpoint_requires_complete_in_either_layout(monkeypatch, layout):
    ready = f"/runs/example{layout}/chk000120"
    monkeypatch.setattr(olivia, "text", lambda args: ready + "\n/runs/example/capture/chk000240")
    monkeypatch.setattr(olivia, "ssh", lambda args, **kw: SimpleNamespace(
        returncode=0 if args[-1] == ready + "/rrea_checkpoint/COMPLETE" else 1))
    assert olivia._newest_complete_checkpoint("/runs/example", "auto") == ready
    assert olivia._newest_complete_checkpoint("/runs/example", "chk000120") == ready
    with pytest.raises(SystemExit):
        olivia._newest_complete_checkpoint("/runs/example", "chk000240")


@pytest.mark.parametrize("layout", ["", "capture"])
def test_push_uses_real_checkpoint_location(tmp_path, monkeypatch, layout):
    capture = tmp_path / "capture"
    capture.mkdir()
    for relative in olivia.RESTART_REQUIRED_FILES:
        path = capture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    checkpoint = tmp_path / layout / "chk000120"
    marker = checkpoint / "rrea_checkpoint/COMPLETE"
    marker.parent.mkdir(parents=True)
    marker.touch()
    copied = []
    monkeypatch.setattr(olivia, "ssh", lambda *a, **kw: None)
    monkeypatch.setattr(olivia, "_rsync", lambda source, target: copied.append((source, target)))
    olivia.cmd_push_restart(SimpleNamespace(local_run=str(tmp_path), run="scratch_test", checkpoint="auto"))
    assert copied[0][0] == checkpoint.as_posix() + "/"
    assert copied[0][1].endswith("/runs/scratch_test/chk000120/")


def test_script_forwards_driver_options_verbatim():
    argv = ["script", "submit_gpu.sh", "CPUS=48", "run", "--dt-s", "2.5e-9"]
    args = olivia.build_parser().parse_args(argv)
    assert args.args == ["CPUS=48", "run", "--dt-s", "2.5e-9"]
