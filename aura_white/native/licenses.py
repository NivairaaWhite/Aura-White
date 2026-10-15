"""Who may Aura learn from?  A ledger, not a hope.

Training a model on another model's output, or on someone's 3-D assets, is only as clean as the licences behind it.
This module is the single place that decides, and every shard / checkpoint Aura writes carries the decision.

Rules (deliberately conservative):
  * an entry is usable as a TEACHER only if its licence was read and allows using outputs to train other models;
  * anything not listed, or listed as ``unverified``, is DENIED unless the caller passes ``accept_risk=True`` (the
    refusal says why, and the acceptance is written into the provenance of everything produced);
  * per-asset licences of datasets (Objaverse-style) are filtered, not assumed: CC0 and CC-BY pass in commercial mode,
    non-commercial variants only in research mode, anything flagged "no AI" never.

This is engineering bookkeeping, not legal advice.  "verified" lines record the date and the page that was read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# status: ok | restricted | denied | unverified
@dataclass(frozen=True)
class Source:
    key: str
    name: str
    licence: str
    status: str                     # ok | noncommercial | denied | unverified
    why: str
    verified: str = ""              # "2026-10-13 github.com/microsoft/TRELLIS.2" etc.; empty = not read by us
    attribution: str = ""


MODELS: Dict[str, Source] = {s.key: s for s in [
    Source("trellis2", "TRELLIS.2 (Microsoft)", "MIT (nvdiffrast / nvdiffrec dependencies carry their own licences)", "ok",
           "MIT licence on code and weights; training code released.", "2026-10-13 github.com/microsoft/TRELLIS.2",
           "TRELLIS.2, Microsoft, MIT"),
    Source("triposg", "TripoSG (VAST-AI-Research)", "MIT", "ok", "MIT licence; 1.5 B rectified-flow shape model.",
           "2026-10-13 github.com/VAST-AI-Research/TripoSG", "TripoSG, VAST-AI-Research, MIT"),
    Source("triposr", "TripoSR (Stability AI / Tripo)", "MIT", "ok", "MIT licence (code and weights).",
           "NOTICE.md of Aura White 7", "TripoSR, Stability AI and Tripo AI, MIT"),
    Source("hunyuan3d-2", "Hunyuan3D 2.0 / 2mv (Tencent)", "Tencent Hunyuan 3D 2.0 Community License", "denied",
           "Section 5(b): you must not use the Works or any Output or results to improve any other AI model; the licence also "
           "excludes the EU, UK and South Korea.", "2026-10-13 huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE"),
    Source("hunyuan3d-2.1", "Hunyuan3D 2.1 (Tencent)", "Tencent Hunyuan 3D 2.1 Community License", "denied",
           "Section 5(b) forbids using Output to improve any other AI model; territory excludes the EU, UK and South Korea.",
           "2026-10-13 github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE"),
    Source("zero123plus", "Zero123++ (SUDO-AI)", "code Apache-2.0, weights CC-BY-NC 4.0", "noncommercial",
           "Weights are non-commercial: a student trained on its views inherits the restriction.", "NOTICE.md of Aura White 7"),
    Source("instantmesh", "InstantMesh (Tencent ARC)", "Apache-2.0 (per its README)", "unverified",
           "Not re-read this session; its pipeline also relies on Zero123++ style multi-view generation.", ""),
    Source("pixal3d", "Pixal3D", "MIT (reported in a 2026 roundup)", "unverified", "Not read at the source by us.", ""),
    Source("step1x-3d", "Step1X-3D", "Apache-2.0 (reported)", "unverified", "Not read at the source by us.", ""),
    Source("sf3d", "Stable Fast 3D (Stability AI)", "Stability AI Community License (reported)", "unverified",
           "Community licences usually carry revenue conditions; not read by us.", ""),
    Source("meshy", "Meshy (SaaS)", "service terms", "unverified", "Hosted service: its terms, not an open licence, decide; free tier "
           "outputs are reported to be CC BY 4.0 with a monthly credit limit; reports conflict.", ""),
    Source("tripo", "Tripo (SaaS)", "service terms", "unverified", "Hosted service; free tier outputs reported public / "
           "non-commercial.", ""),
    Source("rodin", "Rodin (SaaS)", "service terms", "unverified", "Hosted service; read its terms.", ""),
    Source("own", "your own assets / procedural generator", "yours", "ok", "Assets you made or generated with Aura's own code.", ""),
]}

# Names people actually type.  A teacher key is canonicalised (case, '_' / ' ' / '.' spelling) and mapped onto the ledger, and any
# name that mentions a denied family resolves to that family: "Hunyuan3D_2.5" or "hunyuan3d-2mv" must not slip through as
# "not in the ledger" (which accept_risk would then allow).
_ALIASES: Dict[str, str] = {
    "trellis.2": "trellis2", "trellis-2": "trellis2", "microsoft-trellis2": "trellis2",
    "tripo-sg": "triposg", "tripo-sr": "triposr",
    "zero123++": "zero123plus", "zero123-plus": "zero123plus", "zero123pp": "zero123plus",
    "hunyuan3d-2mv": "hunyuan3d-2", "hunyuan3d-2.0": "hunyuan3d-2", "hunyuan3d-2-mini": "hunyuan3d-2",
    "hunyuan3d-2.1": "hunyuan3d-2.1", "hunyuan3d-21": "hunyuan3d-2.1",
}
_DENIED_FAMILIES = ("hunyuan",)            # any key containing one of these is treated as the (denied) Hunyuan entry


def _canon(key: str) -> str:
    return re.sub(r"[\s_]+", "-", (key or "").strip().lower())


def resolve(key: str) -> Optional["Source"]:
    """Ledger entry for a teacher name, or None.  Never returns an 'ok' entry for a name that merely contains an
    'ok' name; denied families win over everything."""
    k = _canon(key)
    if any(f in k for f in _DENIED_FAMILIES):
        return MODELS["hunyuan3d-2.1"] if ("2.1" in k or "21" in k) else MODELS["hunyuan3d-2"]
    k = _ALIASES.get(k, k)
    return MODELS.get(k)


# Dataset-level per-asset licence tags (Objaverse style: 'by', 'cc0', 'by-nc' ...)
_COMMERCIAL_OK = {"cc0", "by", "cc-by", "cc-by-4.0", "cc0-1.0", "public-domain", "pd", "mit", "apache-2.0", "own"}
_RESEARCH_EXTRA = {"by-sa", "cc-by-sa", "by-nc", "cc-by-nc", "by-nc-sa", "cc-by-nc-sa", "standard", "cc-by-sa-4.0"}
_NEVER = {"by-nd", "cc-by-nd", "by-nc-nd", "cc-by-nc-nd", "all-rights-reserved", "noai", "no-ai", "unknown", ""}


class LicenceError(RuntimeError):
    pass


@dataclass
class Decision:
    allowed: bool
    reason: str
    source: str
    risk_accepted: bool = False

    def as_dict(self):
        return asdict(self)


def can_teach(key: str, commercial: bool = True, accept_risk: bool = False) -> Decision:
    """May the output of model `key` be used to train Aura?"""
    s = resolve(key)
    if s is None:
        if accept_risk:
            return Decision(True, f"'{key}' is not in the ledger; risk accepted by the caller", key, True)
        return Decision(False, f"'{key}' is not in the licence ledger - add it to aura_white/native/licenses.py after reading its "
                               f"licence, or pass accept_risk=True", key)
    if s.status == "ok":
        return Decision(True, s.why, key)
    if s.status == "noncommercial":
        if not commercial:
            return Decision(True, s.why + " (allowed because this run is non-commercial research)", key)
        return Decision(False, s.why, key)
    if s.status == "denied":
        return Decision(False, s.why, key)             # no override: a licence that forbids it is not a risk to "accept"
    # unverified
    if accept_risk:
        return Decision(True, s.why + " Risk accepted by the caller.", key, True)
    return Decision(False, s.why + " Denied by default until the licence is read; pass accept_risk=True to override.", key)


def require_teacher(key: str, commercial: bool = True, accept_risk: bool = False) -> Decision:
    d = can_teach(key, commercial, accept_risk)
    if not d.allowed:
        raise LicenceError(f"cannot use {key!r} as a teacher: {d.reason}")
    return d


def asset_licence_ok(tag: Optional[str], commercial: bool = True, no_ai_flag: bool = False) -> Tuple[bool, str]:
    """Filter for a dataset asset's own licence tag."""
    if no_ai_flag:
        return False, "creator flagged the asset 'no AI training'"
    t = (tag or "").strip().lower()
    if t in _NEVER:
        return False, f"licence '{t or 'missing'}' does not permit this use"
    if t in _COMMERCIAL_OK:
        return True, t
    if t in _RESEARCH_EXTRA:
        return (False, f"licence '{t}' is only accepted in research (non-commercial) mode") if commercial else (True, t)
    return False, f"unknown licence tag '{t}'"


def ledger_table() -> List[Dict]:
    return [dict(key=s.key, name=s.name, licence=s.licence, status=s.status, why=s.why, verified=s.verified) for s in MODELS.values()]


@dataclass
class Provenance:
    """Written into every shard manifest and every checkpoint card."""
    sources: Dict[str, int] = field(default_factory=dict)             # source key -> number of records
    decisions: Dict[str, Dict] = field(default_factory=dict)
    attributions: List[str] = field(default_factory=list)
    commercial: bool = True

    def add(self, key: str, n: int, decision: Optional[Decision] = None, attribution: str = ""):
        self.sources[key] = self.sources.get(key, 0) + int(n)
        if decision is not None:
            self.decisions[key] = decision.as_dict()
        if attribution and attribution not in self.attributions:
            self.attributions.append(attribution)

    def merge(self, other: "Provenance"):
        for k, n in other.sources.items():
            self.sources[k] = self.sources.get(k, 0) + n
        self.decisions.update(other.decisions)
        for a in other.attributions:
            if a not in self.attributions:
                self.attributions.append(a)
        self.commercial = self.commercial and other.commercial

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "Provenance":
        return Provenance(sources=dict(d.get("sources", {})), decisions=dict(d.get("decisions", {})),
                          attributions=list(d.get("attributions", [])), commercial=bool(d.get("commercial", True)))

    def commercially_clean(self) -> Tuple[bool, List[str]]:
        problems = []
        for k, dec in self.decisions.items():
            if dec.get("risk_accepted"):
                problems.append(f"{k}: licence risk was accepted manually")
            s = resolve(k)
            if s is not None and s.status in ("noncommercial", "denied"):
                problems.append(f"{k}: {s.status}")
        return (not problems), problems
