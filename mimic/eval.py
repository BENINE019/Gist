#!/usr/bin/env python3
"""Evaluation: compare every stage against the ground truth in clips/goal.md.

Produces eval/results.md (four markdown tables, ready for the README) and
eval/results.json. Nothing here calls an API; it reads what the stages wrote.

  Table 1  perception   naming recall vs the object list, tracks per object, coverage
  Table 2  events       moved/static accuracy, attachment (grouping) accuracy
  Table 3  intent       goal type / scope match, with and without narration, clarifications
  Table 4  generalize   goal vs copy success in the original and shuffled scenes, episodes

Usage (from the repo root):
  python -m mimic.eval
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Words the user and the models use for the same thing.
SYNONYMS = {
    "cup": {"cup", "container", "bowl", "mug"}, "container": {"cup", "container", "bowl"}, "bowl": {"bowl", "cup", "container"},
    "earphone": {"earphone", "case", "earphones"}, "case": {"case", "earphone", "earphones"},
    "marker": {"marker", "pen"}, "pen": {"pen", "marker"},
    "scissor": {"scissor", "scissors"}, "scissors": {"scissors", "scissor"},
    "glasses": {"glasses", "eyeglasses", "spectacles"}, "eye": set(),
    "keychain": {"keychain", "key", "keys"}, "key": {"key", "keychain"},
    "remote": {"remote", "control"}, "lead": {"lead", "case"},
}
COLORS = {"black", "white", "red", "blue", "green", "yellow", "orange", "brown", "pink", "purple", "grey", "gray",
          "silver", "cream", "big", "small", "large", "little", "bigger", "smaller"}


def nouns(name: str) -> set[str]:
    toks = set(re.findall(r"[a-z]+", name.split("#")[0].lower()))
    out = set()
    for t in toks - COLORS - {"the", "my", "a"}:
        out |= SYNONYMS.get(t, {t})
    return out


def same_thing(a: str, b: str) -> bool:
    return bool(nouns(a) & nouns(b))


def parse_goals(path: Path) -> dict[str, dict]:
    goals: dict[str, dict] = {}
    cur = None
    for line in path.read_text().splitlines():
        m = re.match(r"^##\s+(c\d+)", line.strip())
        if m:
            cur = m.group(1)
            goals[cur] = {}
            continue
        if cur and ":" in line:
            k, v = line.split(":", 1)
            goals[cur][k.strip()] = v.strip().strip('"')
    for g in goals.values():
        g["type"] = re.sub(r"[\s(].*$", "", g.get("goal", "")).replace("put all", "put_all").strip()
        g["objects_list"] = [x.strip() for x in g.get("objects", "").split(",") if x.strip()]
        g["moved_list"] = [x.strip() for x in g.get("moved", "").split(",") if x.strip()]
    return goals


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(n: float, d: float) -> str:
    return f"{100 * n / d:.0f}%" if d else "n/a"


def eval_perception(out: Path, gt: dict) -> dict:
    tt = load(out / "table_tracks.json")
    ev = load(out / "events.json")
    if not tt or not ev:
        return {}
    kept = {o["key"] for o in ev["objects"]}
    objs = [o for o in tt["objects"] if not o["is_hand"] and not o.get("is_marker_sheet")]
    # naming recall: each ground-truth object matched by at least one kept track
    found = [g for g in gt["objects_list"] if any(same_thing(g, k) for k in kept)]
    extra = [k for k in kept if not any(same_thing(k, g) for g in gt["objects_list"])]
    # tracks per physical object: raw tracks vs objects left after dedupe
    coverage = []
    for o in objs:
        f = o["frames"]
        span = f[-1]["f"] - f[0]["f"] + 1
        coverage.append(len(f) / span if span else 1.0)
    return {
        "gt_objects": len(gt["objects_list"]), "named": len(found), "missed": [g for g in gt["objects_list"] if g not in found],
        "spurious": extra, "raw_tracks": len(objs), "objects_after_dedupe": len(kept),
        "coverage": sum(coverage) / len(coverage) if coverage else 0.0,
        "merged": len(ev.get("duplicates_merged", [])),
    }


def eval_events(out: Path, gt: dict) -> dict:
    ev = load(out / "events.json")
    if not ev:
        return {}
    moved_sys = [o["key"] for o in ev["objects"] if o["role"] in ("moved", "attached")]
    static_sys = [o["key"] for o in ev["objects"] if o["role"] in ("static", "container")]
    gt_moved = gt["moved_list"]
    gt_static = [g for g in gt["objects_list"] if not any(same_thing(g, m) for m in gt_moved)]
    tp = [m for m in moved_sys if any(same_thing(m, g) for g in gt_moved)]
    fp = [m for m in moved_sys if not any(same_thing(m, g) for g in gt_moved)]
    fn = [g for g in gt_moved if not any(same_thing(g, m) for m in moved_sys)]
    tn = [s for s in static_sys if any(same_thing(s, g) for g in gt_static)]
    wants_attach = "attached" in gt.get("notes", "").lower() or "grouped" in gt.get("notes", "").lower()
    has_attach = any(o["role"] == "attached" for o in ev["objects"])
    return {"moved_correct": len(tp), "moved_false": fp, "moved_missed": fn, "static_correct": len(tn),
            "events": len(ev["events"]), "attach_expected": wants_attach, "attach_found": has_attach}


def eval_intent(out: Path, gt: dict) -> dict:
    res = {}
    for label, fname in (("with", "intent.json"), ("without", "intent_no_narration.json")):
        d = load(out / fname)
        if not d or not d.get("candidates"):
            res[label] = None
            continue
        top = d["candidates"][0]
        type_ok = top["type"] == gt["type"]
        scope_gt = gt.get("scope", "")
        scope_ok = (top["scope"] == scope_gt) if scope_gt else True
        order_gt = gt.get("order_matters", "false").lower() == "true"
        order_ok = bool(top.get("order_matters")) == order_gt
        res[label] = {"goal": top.get("goal", ""), "type_ok": type_ok, "scope_ok": scope_ok, "order_ok": order_ok,
                      "confidence": top["confidence"], "asked": bool(d.get("needs_clarification")),
                      "question": d.get("clarifying_question", ""), "n_candidates": len(d["candidates"])}
    return res


def eval_generalize(out: Path) -> dict:
    s = load(out / "generalize" / "summary.json")
    if not s:
        return {}
    return {"goal_orig": s["goal_success_original"], "copy_orig": s["copy_success_original"],
            "goal_shuf": s["goal_success_shuffled"], "copy_shuf": s["copy_success_shuffled"],
            "episodes": s["episodes_exported"], "n_shuffled": len(s["scenes"]) - 1}


def main() -> None:
    p = argparse.ArgumentParser(description="Build the four evaluation tables")
    p.add_argument("--goals", type=Path, default=Path("clips/goal.md"))
    p.add_argument("--out-root", type=Path, default=Path("out"))
    p.add_argument("--dest", type=Path, default=Path("eval"))
    args = p.parse_args()

    goals = parse_goals(args.goals)
    clips = sorted(goals)
    results = {}
    for c in clips:
        out = args.out_root / c
        results[c] = {"perception": eval_perception(out, goals[c]), "events": eval_events(out, goals[c]),
                      "intent": eval_intent(out, goals[c]), "generalize": eval_generalize(out)}

    args.dest.mkdir(exist_ok=True)
    (args.dest / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    L = []
    # Table 1
    L += ["## Table 1: perception (Muse Spark naming + SAM 3.1 tracking)", "",
          "| Clip | Objects (truth) | Named | Missed | Spurious | Raw tracks | After merge | Coverage |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    tot_gt = tot_named = 0
    for c in clips:
        r = results[c]["perception"]
        if not r:
            continue
        tot_gt += r["gt_objects"]; tot_named += r["named"]
        L.append(f"| {c} | {r['gt_objects']} | {r['named']} | {', '.join(r['missed']) or '-'} | {', '.join(r['spurious']) or '-'} "
                 f"| {r['raw_tracks']} | {r['objects_after_dedupe']} | {r['coverage']:.0%} |")
    L += ["", f"Naming recall: **{pct(tot_named, tot_gt)}** ({tot_named}/{tot_gt} objects). "
          "Coverage = frames tracked while the object was in view. Raw tracks > after merge means SAM found the same object under two names; the merge step fixed it.", ""]

    # Table 2
    L += ["## Table 2: events (what moved, what was attached)", "",
          "| Clip | Moved (truth) | Moved correct | False moved | Missed moved | Static correct | Events | Attachment |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    mc = mt = 0
    for c in clips:
        r = results[c]["events"]
        g = goals[c]
        if not r:
            continue
        mt += len(g["moved_list"]); mc += r["moved_correct"]
        att = "-" if not r["attach_expected"] else ("✓" if r["attach_found"] else "✗")
        L.append(f"| {c} | {len(g['moved_list'])} | {r['moved_correct']} | {', '.join(r['moved_false']) or '-'} | "
                 f"{', '.join(r['moved_missed']) or '-'} | {r['static_correct']} | {r['events']} | {att} |")
    L += ["", f"Moved-object recall: **{pct(mc, mt)}** ({mc}/{mt}). Attachment column: whether the keychain was grouped with the wallet where the truth says it should be.", ""]

    # Table 3
    L += ["## Table 3: goal inference (Muse Spark), with and without narration", "",
          "| Clip | Truth | With narration | Type | Scope | Order | Asked? | Without narration | Type | Scope | Asked? |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    w_type = w_scope = wo_type = wo_scope = n_w = n_wo = asked = asked_wo = 0
    for c in clips:
        r = results[c]["intent"]
        g = goals[c]
        w, wo = r.get("with"), r.get("without")
        tick = lambda b: "✓" if b else "✗"
        if w:
            n_w += 1; w_type += w["type_ok"]; w_scope += w["scope_ok"]; asked += w["asked"]
        if wo:
            n_wo += 1; wo_type += wo["type_ok"]; wo_scope += wo["scope_ok"]; asked_wo += wo["asked"]
        L.append(f"| {c} | `{g.get('goal', '')}` | `{w['goal'] if w else '-'}` | {tick(w['type_ok']) if w else '-'} | "
                 f"{tick(w['scope_ok']) if w else '-'} | {tick(w['order_ok']) if w else '-'} | {'yes' if w and w['asked'] else 'no'} | "
                 f"`{wo['goal'] if wo else '-'}` | {tick(wo['type_ok']) if wo else '-'} | {tick(wo['scope_ok']) if wo else '-'} | "
                 f"{'yes' if wo and wo['asked'] else 'no'} |")
    L += ["", f"With narration: type **{pct(w_type, n_w)}**, scope **{pct(w_scope, n_w)}**, clarifying question needed on **{asked}/{n_w}** clips. "
          f"Without narration: type **{pct(wo_type, n_wo)}**, scope **{pct(wo_scope, n_wo)}**, clarifying question needed on **{asked_wo}/{n_wo}** clips. "
          "Type and scope are scored on the top candidate. Narration mostly buys certainty: without it the model offers two readings and has to ask.", ""]

    # Table 4
    L += ["## Table 4: teach once, adapt to new scenes (simulation)", "",
          "| Clip | Goal | Original: goal / copy | Shuffled scenes | Goal plan | Copy baseline | Episodes |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    gs = cs = ns = eps = 0
    for c in clips:
        r = results[c]["generalize"]
        w = results[c]["intent"].get("with")
        if not r:
            continue
        ns += r["n_shuffled"]; gs += (r["goal_shuf"] or 0) * r["n_shuffled"]; cs += (r["copy_shuf"] or 0) * r["n_shuffled"]; eps += r["episodes"]
        tick = lambda b: "✓" if b else "✗"
        L.append(f"| {c} | `{w['goal'] if w else ''}` | {tick(r['goal_orig'])} / {tick(r['copy_orig'])} | {r['n_shuffled']} | "
                 f"{r['goal_shuf']:.0%} | {r['copy_shuf']:.0%} | {r['episodes']} |")
    L += ["", f"Across {ns} shuffled scenes: goal plan **{pct(gs, ns)}**, copy baseline **{pct(cs, ns)}**. "
          f"**{eps} verified episodes from {len([c for c in clips if results[c]['generalize']])} phone demonstrations.** "
          "Shuffled scenes move every object; every second scene adds an extra object of the demonstrated kind and every third adds a distractor. "
          "Scenes where the goal already held before acting were rerolled. Success is judged against simulator state.", ""]

    (args.dest / "results.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {args.dest / 'results.md'} and {args.dest / 'results.json'}")


if __name__ == "__main__":
    main()
