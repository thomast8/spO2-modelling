"""Contract §17: lung packing, measured never assumed.

Efficiency e = (packedVC − baseVC) / packs from the profile's measurements; a packed hold adds
ΔV = e · min(packCount, packs) litres, nothing when its packCount is not recorded, and nothing at
all without a packed measurement. The hold model removes what the extra air should buy with a
fixed offset ln(U(V_full + ΔV) / U(V_full)), U being the usable oxygen store at the profile.
"""

import math

from archive import finished, is_performed_hold
from evidence import evaluate
from features import hold_model_rows
from fit_hold_model import fit
from fit_oxygen_store import stores, volumes_for


def efficiency(profile):
    """Litres per pack, or None without a packed measurement and a VC (else FVC) to compare."""
    measured = (profile or {}).get("measured", {})
    packed, packs = measured.get("packedVC"), measured.get("packs")
    base = measured.get("vc", measured.get("fvc"))
    if packed is None or base is None or not packs:
        return None
    return (packed - base) / packs


def extra_volume(e, pack_count, packs):
    """ΔV in litres, capped at the measured pack count so it never extrapolates."""
    if e is None or pack_count is None:
        return 0.0
    return e * min(pack_count, packs)


def usable(constants, profile, spiro, ratios, extra=0.0):
    """Usable O₂ store (mL) at full lung plus `extra` litres, or None when the profile cannot
    give it."""
    vols = volumes_for(profile, spiro, ratios)
    if vols is None or profile.get("weightKg") is None:
        return None
    return stores(constants, profile, {"full": vols["full"][0] + extra})["usable"]["full"]


class Packing:
    """What packing adds for one profile; `offset(step)` is 0 whenever anything is missing."""

    def __init__(self, profile, constants, spiro, ratios):
        self.e = efficiency(profile) if profile and constants else None
        self.packs = ((profile or {}).get("measured") or {}).get("packs")
        self.profile, self.constants, self.spiro, self.ratios = profile, constants, spiro, ratios
        self.base = usable(constants, profile, spiro, ratios) if self.e is not None else None
        if self.base is None:
            self.e = None

    @property
    def measured(self):
        return self.e is not None

    def delta_v(self, pack_count):
        return extra_volume(self.e, pack_count, self.packs)

    def offset_for(self, pack_count):
        dv = self.delta_v(pack_count)
        if not dv:
            return 0.0
        return math.log(
            usable(self.constants, self.profile, self.spiro, self.ratios, dv) / self.base
        )

    def offset(self, step):
        """The §17 offset for a hold step: packed holds only."""
        if step.get("style") != "packed":
            return 0.0
        return self.offset_for(step.get("packCount"))


NONE = Packing(None, None, None, None)


def packed_gate(full_pb_gate, packing):
    """§17: the packed PB estimate's gate is the full-lung pb gate plus the packed measurement."""
    inputs = full_pb_gate.inputs
    floors = {
        **inputs["floors"],
        "enough": {**inputs["floors"]["enough"], "lungVolume": 1},
        "keep": {**inputs["floors"]["keep"], "lungVolume": 1},
    }
    counts = {tuple(k.split(".", 1)): c for k, c in inputs["counts"].items()}
    counts[("lungVolume", "packedVC")] = int(packing.measured)
    n = inputs["narrowing"]
    narrowing = None if n is None else (n["realised"], n["rowsToReach"])
    return evaluate("pb.packed", floors, counts, narrowing, inputs["previouslyShown"])


def _clock(t):
    return f"{int(t) // 60}:{round(t) % 60:02d}"


def report(sessions, out, packing, hold_model, physiology=None, tz=None):
    """Section 4b: the packed holds, what packing adds for the profile (if any), the packed PB."""
    out.append("4b. PACKING (contract §17)")
    out.append("-" * 26)
    out.append("")
    out.append(
        "Packed holds count in the full-lung slot. What packing adds is measured, never assumed: "
        "it needs\na packed vital capacity and its pack count, and a hold adds air only for its "
        "recorded packCount."
    )
    out.append("")
    out.append(f"  {'session':<12}{'packs':<14}{'held':>6}  {'outcome':<12}row")
    holds = []
    for s in finished(sessions):
        rows = {r["stepIndex"]: r for r in hold_model_rows(s)}
        for i, st in enumerate(s["steps"]):
            if not (is_performed_hold(st) and st.get("style") == "packed"):
                continue
            o = st["outcome"]
            outcome = o.get("reason") or o["type"]
            row = rows.get(i)
            status = "not a row" if row is None else "censored" if row["censored"] else "UNCENSORED"
            packs = "not recorded" if st.get("packCount") is None else str(st["packCount"])
            holds.append((s, i, st, row))
            out.append(
                f"  {s['startedAt'][:10]:<12}{packs:<14}{_clock(st['actualDuration']):>6}  "
                f"{outcome:<12}{status}"
            )
    out.append("")
    model, cfg = hold_model["model"], hold_model["config"]
    est = hold_model["estimates"]
    if not packing.measured:
        out.append(
            "  No packed measurement: packing adds nothing (no efficiency, every ΔV and offset is "
            "0), so the\n  full-lung numbers above are unchanged by it."
        )
    else:
        m = packing.profile["measured"]
        base = m.get("vc", m.get("fvc"))
        out.append(
            f"  Packing efficiency e = ({m['packedVC']} − {base}) / {m['packs']} = "
            f"{packing.e:.3f} L per pack (a measurement, never scored)."
        )
        full_store = packing.base
        top = usable(
            packing.constants,
            packing.profile,
            packing.spiro,
            packing.ratios,
            packing.delta_v(m["packs"]),
        )
        out.append(
            f"  Usable O₂ store: full lung {full_store / 1000:.2f} L, at {m['packs']} packs "
            f"{top / 1000:.2f} L."
        )
        out.append(
            f"    {'session':<12}{'packs':<14}{'ΔV L':>6}{'offset':>8}{'store':>8}{'of hold':>9}"
        )
        for s, _, st, _ in holds:
            dv = packing.delta_v(st.get("packCount"))
            off = packing.offset(st)
            out.append(
                f"    {s['startedAt'][:10]:<12}"
                f"{'not recorded' if st.get('packCount') is None else st['packCount']!s:<14}"
                f"{dv:>6.2f}{off:>8.3f}{100 * (math.exp(off) - 1):>7.1f}%"
                f"{st['actualDuration'] * (1 - math.exp(-off)):>7.0f} s"
            )
        out.append(
            "    offset = ln(U(V_full + ΔV) / U(V_full)); 'store' is what the extra air adds to "
            "the usable store,\n    'of hold' the seconds the model removes from that hold."
        )
        plain = fit(sessions, cfg, physiology, tz)
        a = math.exp(model["post"]["full"][0]) if "full" in model["post"] else None
        b = math.exp(plain["post"]["full"][0]) if "full" in plain["post"] else None
        if a and b:
            out.append(
                f"  Full-lung level centre with the packing offsets {_clock(a)}, without "
                f"{_clock(b)} ({a - b:+.0f} s)."
            )
    gate = packed_gate(est["full"]["pb"]["gate"], packing)
    if gate.enough and est["full"]["pb"]["estimate"]:
        uncensored = [
            st.get("packCount")
            for _, _, st, row in holds
            if row is not None and not row["censored"] and st.get("packCount") is not None
        ]
        k = max(sorted(set(uncensored)), key=uncensored.count) if uncensored else packing.packs
        off = packing.offset_for(k)
        lo, mid, hi = (x * math.exp(off) for x in est["full"]["pb"]["estimate"])
        out.append(f"  Packed PB at {k} packs: {_clock(lo)}-{_clock(hi)} (centre {_clock(mid)}).")
    else:
        out.append(
            f"  Packed PB estimate (shown only next to a ready full-lung one): {gate.summary}"
        )
    out.append("")
    return gate
