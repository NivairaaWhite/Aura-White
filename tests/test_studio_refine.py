"""Integration tests for the studio's repair / snap / completion / best-of pipeline stages."""
import pytest
from synth import FakeGeometry, make_artwork, make_object

from aura_white.studio import AuraStudio, StudioOptions

OPTS = dict(quality="draft", texture_size=512, target_faces=4000, normal_map=False, multiview="off", fidelity="standard")


@pytest.fixture
def artwork():
    v, f = make_object(2)
    from aura_white.raster import get_raster

    img = make_artwork(get_raster("cpu"), v, f, size=768)
    return v, f, img


def test_repair_snap_complete_run_without_crashing_and_report(tmp_path, artwork, monkeypatch):
    v, f, img = artwork
    p = tmp_path / "art.png"
    img.save(p)
    st = AuraStudio(options=StudioOptions(geometry="fake", **OPTS, repair=True, snap=True, complete=True))
    monkeypatch.setattr(st, "_geometry_backend", lambda o: FakeGeometry(v.copy(), f.copy()))
    res = st.generate(str(p))
    assert "repair" in res.report["refine"]
    assert res.report["refine"]["repair"]["watertight"] in (True, False)
    assert len(res.faces) > 0


def test_completion_fills_a_hole_the_geometry_backend_missed(tmp_path, artwork, monkeypatch):
    v, f, img = artwork
    p = tmp_path / "art.png"
    img.save(p)
    # drop every face touching the +z tip (a torus decoration) so the neural mesh "misses" a part the artwork shows
    tip = v[f].mean(1)[:, 2] < 0.55
    v2, f2 = v.copy(), f[tip]
    st_no = AuraStudio(options=StudioOptions(geometry="fake", **OPTS, repair=False, snap=False, complete=False))
    st_yes = AuraStudio(options=StudioOptions(geometry="fake", **OPTS, repair=False, snap=False, complete=True))
    monkeypatch.setattr(st_no, "_geometry_backend", lambda o: FakeGeometry(v2.copy(), f2.copy()))
    monkeypatch.setattr(st_yes, "_geometry_backend", lambda o: FakeGeometry(v2.copy(), f2.copy()))
    r_no = st_no.generate(str(p))
    r_yes = st_yes.generate(str(p))
    assert r_yes.report["refine"].get("completion", {}).get("faces", 0) > 0
    assert r_yes.report["reference_iou"] >= r_no.report["reference_iou"]


class _StatusFakeGeometry(FakeGeometry):
    def status(self):
        return True, "fake"


def test_best_of_picks_the_better_matching_geometry(tmp_path, artwork, monkeypatch):
    v, f, img = artwork
    p = tmp_path / "art.png"
    img.save(p)
    # "bad" is missing the torus decoration entirely -> a genuinely worse silhouette match, not just a rescaling
    # (registration normalises scale away, so a scale-only perturbation would score identically to the original)
    keep = v[f].mean(1)[:, 2] < 0.55
    bad_v, bad_f = v.copy(), f[keep]
    st = AuraStudio(options=StudioOptions(**OPTS, best_of=("good", "bad"), complete=False))

    from aura_white.backends import geometry as geo_mod

    def fake_make(name, device, **kw):
        return _StatusFakeGeometry(v.copy(), f.copy()) if name == "good" else _StatusFakeGeometry(bad_v, bad_f)

    monkeypatch.setattr(geo_mod, "make_geometry", fake_make)
    res = st.generate(str(p))
    notes = "\n".join(res.report["notes"])
    assert "best_of silhouette scores" in notes
    scores = dict(pair.split("=") for pair in
                 notes.split("best_of silhouette scores:")[1].split("->")[0].strip().split(", "))
    assert float(scores["good"]) > float(scores["bad"]) + 0.02
    assert res.verts[:, 2].max() > 0.7                        # the chosen mesh kept the torus, so "good" was used
