#!/usr/bin/env python3
"""
reliability_probe_v2.py — v2 feasibility probe for the Agent Reliability project.

WHY v2 (see Design §15 + Brief Decisions log, 19 Jun 2026)
----------------------------------------------------------
v1 (`reliability_probe.py`) SATURATED: flat success=1.00 to N=20. Two causes:
  (a) reset-per-step context — each sub-task ran in a fresh conversation, so "N=20"
      was 20 independent easy tasks, not one long-horizon task (nothing to lose track of);
  (b) at-ceiling trivial steps — 3-line functions, nothing to propagate.

v2 fixes BOTH and isolates the cause:
  1. ACCUMULATING CONVERSATION — one continuous context across all N steps. The agent's
     OWN prior turns are its only window into what it built (no signature crutch is
     re-injected in `accumulate` mode), so "losing the thread" in a growing transcript
     becomes possible. Cost is controlled with rolling prompt caching on the growing
     prefix (cache reads ~10% of base input; Design §14).
  2. HARDER, MORE-COUPLED STEPS — a deep ledger spine where later ops compose earlier
     ops AND must respect conventions the agent established many turns earlier WITHOUT
     those conventions being restated. An early slip cascades; a long transcript invites
     mis-recall. Each step is still at-ceiling *in isolation* (calibration provides the
     conventions + seeds upstream), which is what licenses the reliability-not-capability
     reading (Design §2, §9).

The A/B knob: `--context-mode {accumulate,reset}` runs the SAME v2 ops either as one
growing conversation (accumulate) or fresh-context-per-step with conventions + signatures
re-handed each step (reset). Holding the steps constant, `reset` is the control: if
`accumulate` bends and `reset` stays flat, accumulated context is the cause (Design §15).

CONSTRUCT-VALIDITY NOTE (flag for John): in `accumulate` mode the conventions a recall-op
needs are NOT in its prompt — they are in the agent's earlier turns (it wrote them itself).
That is the thread-keeping test, not spec ambiguity: the info is present in-context, just
earlier. The §9 position-controlled falsifier (reference-correct upstream IN the transcript,
vary position k) is what separates "context degradation" from "propagation" and from "the
per-step task is just harder". Calibration provides the conventions explicitly, so isolation
p̂ measures marginal capability; the chain withholds them from the prompt (they live in
history), so chain-below-null is the reliability signal. See Brief Decisions log.

SCAFFOLD (Design §5): native tool-calling loop, tools = append_function / run_tests /
done / abstain + a per-step max-step cap. Abstain ENABLED in chains, DISABLED in calibration.

SAFETY: executes model-generated Python in a subprocess with a timeout, in a temp dir.
Tasks are trivial but generated code is still untrusted — for the full study consider a
container. Do not run on a machine with sensitive data without sandboxing.

USAGE
-----
  cp .env.example .env   # then paste ANTHROPIC_API_KEY into .env  (gitignored)
  python reliability_probe_v2.py --provider anthropic --model claude-haiku-4-5-20251001 \
         --context-mode accumulate --Ns 4,6,8,10,12,14,16 --runs 10 --calib 20 --cache

  # Offline pipeline validation, zero API cost, no key:
  python reliability_probe_v2.py --mock --Ns 4,8,12 --runs 3 --calib 3
  python reliability_probe_v2.py --mock --mock-bug add_entry --Ns 8 --runs 2 --calib 0

Outputs: results_<ts>.jsonl (per-run records) + calib_<ts>.jsonl (per-op p̂) + a summary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, cast

# Pure, validated helpers reused from v1 (no API/caching concerns; v1 is left untouched).
# grade_step/run_visible are wrapped below to accept Path + V2Op cleanly (v1 types them
# for str + Op, but they only read attributes V2Op also has — structurally compatible).
from reliability_probe import (
    grade_step as _grade_step,
    run_visible as _run_visible,
    sigs,
    TOOLS,
    to_openai_tools,
)


# --------------------------------------------------------------------------- #
# Zero-dependency .env loader. Precedence: a variable already set in the shell
# WINS (so secret managers / CI are never overridden); .env only fills the gaps.
# --------------------------------------------------------------------------- #
def load_dotenv(path: str | None = None) -> None:
    p = Path(path) if path else Path(__file__).with_name(".env")
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


# --------------------------------------------------------------------------- #
# v2 OP BANK — a deeply-coupled ledger spine. A ledger is list[dict] with keys
# 'name' (str) and 'amount' (float).
#
#   spec    : natural-language task. For RECALL ops the spec deliberately does NOT
#             restate the convention (e.g. it says "all fee entries", not "name 'FEE'");
#             the agent must recall what it established earlier.
#   calls   : functions this op INVOKES at runtime — must be seeded for solo calibration.
#   deps    : functions whose CORRECTNESS this op relies on — used for propagation analysis
#             (a superset of `calls`: includes convention-establishers like apply_fee→fee_total).
#   recall  : the established facts a recall-op needs. PROVIDED in calibration/reset (so the
#             step is at-ceiling in isolation); WITHHELD in accumulate (the facts live in the
#             agent's earlier turns). Empty for base/establishing ops.
#   visible : example shown to the agent + run by run_tests().
#   held_out: scored test body — defines check(m)->bool (must `return`). Exact, deterministic.
# --------------------------------------------------------------------------- #
@dataclass
class V2Op:
    name: str
    spec: str
    calls: list[str]
    deps: list[str]
    recall: str
    visible: str
    held_out: str


V2_OP_BANK: list[V2Op] = [
    # --- base layer (no recall) ---
    V2Op("add_entry",
         "Write `add_entry(ledger, name, amount)` that appends {'name': name, 'amount': amount} to the list `ledger` in place and returns None.",
         [], [], "",
         "l=[]\nadd_entry(l,'a',5)\nassert l==[{'name':'a','amount':5}]",
         "l=[]; m.add_entry(l,'a',5); return l==[{'name':'a','amount':5}]"),
    V2Op("total",
         "Write `total(ledger)` returning the sum of every entry's 'amount'. An empty ledger returns 0.0.",
         [], [], "",
         "assert total([])==0.0\nassert total([{'name':'a','amount':2},{'name':'b','amount':3}])==5",
         "return m.total([])==0.0 and m.total([{'name':'a','amount':2},{'name':'b','amount':3}])==5"),
    # --- establishing ops: define the conventions later ops must recall ---
    V2Op("apply_fee",
         "Write `apply_fee(ledger, fee)` that records a fee by CALLING add_entry with name 'FEE' and amount -fee. Returns None.",
         ["add_entry"], ["add_entry"], "",
         "l=[]\napply_fee(l,4)\nassert l==[{'name':'FEE','amount':-4}]",
         "l=[]; m.apply_fee(l,4); return l==[{'name':'FEE','amount':-4}]"),
    V2Op("apply_discount",
         "Write `apply_discount(ledger, disc)` that records a discount by CALLING add_entry with name 'DISC' and amount -disc. Returns None.",
         ["add_entry"], ["add_entry"], "",
         "l=[]\napply_discount(l,3)\nassert l==[{'name':'DISC','amount':-3}]",
         "l=[]; m.apply_discount(l,3); return l==[{'name':'DISC','amount':-3}]"),
    V2Op("apply_tax",
         "Write `apply_tax(ledger, rate)` that appends a tax entry with name 'TAX' and amount -(t * rate), where t is the current total computed by CALLING total(ledger) BEFORE appending. Use add_entry to append. Returns None.",
         ["add_entry", "total"], ["add_entry", "total"], "",
         "l=[{'name':'a','amount':100}]\napply_tax(l,0.5)\nassert l[-1]=={'name':'TAX','amount':-50.0} and len(l)==2",
         "l=[{'name':'a','amount':100}]; m.apply_tax(l,0.5); return l[-1]=={'name':'TAX','amount':-50.0} and len(l)==2"),
    # --- recall ops: spec omits the literal convention; agent must remember it ---
    V2Op("fee_total",
         "Write `fee_total(ledger)` returning the total amount of all fee entries previously recorded in the ledger. An empty ledger returns 0.0.",
         [], ["apply_fee"],
         "Fee entries are recorded under the name 'FEE'.",
         "assert fee_total([{'name':'FEE','amount':-4},{'name':'a','amount':2},{'name':'FEE','amount':-1}])==-5\nassert fee_total([])==0.0",
         "return m.fee_total([{'name':'FEE','amount':-4},{'name':'a','amount':2},{'name':'FEE','amount':-1}])==-5 and m.fee_total([])==0.0"),
    V2Op("discount_total",
         "Write `discount_total(ledger)` returning the total amount of all discount entries previously recorded in the ledger. An empty ledger returns 0.0.",
         [], ["apply_discount"],
         "Discount entries are recorded under the name 'DISC'.",
         "assert discount_total([{'name':'DISC','amount':-3},{'name':'a','amount':5}])==-3\nassert discount_total([])==0.0",
         "return m.discount_total([{'name':'DISC','amount':-3},{'name':'a','amount':5}])==-3 and m.discount_total([])==0.0"),
    V2Op("tax_total",
         "Write `tax_total(ledger)` returning the total amount of all tax entries previously recorded in the ledger. An empty ledger returns 0.0.",
         [], ["apply_tax"],
         "Tax entries are recorded under the name 'TAX'.",
         "assert tax_total([{'name':'TAX','amount':-10.0},{'name':'a','amount':5}])==-10.0\nassert tax_total([])==0.0",
         "return m.tax_total([{'name':'TAX','amount':-10.0},{'name':'a','amount':5}])==-10.0 and m.tax_total([])==0.0"),
    # --- composition ops: reuse the recall ops; deep dependency chains ---
    V2Op("adjustments_total",
         "Write `adjustments_total(ledger)` returning the combined total of every adjustment entry (fees, discounts and tax) by REUSING the three per-category total functions you already wrote (do not re-scan the ledger yourself).",
         ["fee_total", "discount_total", "tax_total"],
         ["fee_total", "discount_total", "tax_total"],
         "The per-category totals are computed by your functions fee_total, discount_total and tax_total.",
         "assert adjustments_total([{'name':'a','amount':100},{'name':'FEE','amount':-4},{'name':'DISC','amount':-3},{'name':'TAX','amount':-10.0}])==-17.0",
         "return m.adjustments_total([{'name':'a','amount':100},{'name':'FEE','amount':-4},{'name':'DISC','amount':-3},{'name':'TAX','amount':-10.0}])==-17.0"),
    V2Op("base_total",
         "Write `base_total(ledger)` returning the total of the ORIGINAL (non-adjustment) entries only — i.e. the full total minus all adjustments — by REUSING your existing functions for those two quantities.",
         ["total", "adjustments_total"],
         ["total", "adjustments_total"],
         "The full sum of all amounts is total(ledger); the combined adjustments are adjustments_total(ledger).",
         "assert base_total([{'name':'a','amount':100},{'name':'FEE','amount':-4},{'name':'DISC','amount':-3},{'name':'TAX','amount':-10.0}])==100.0",
         "return m.base_total([{'name':'a','amount':100},{'name':'FEE','amount':-4},{'name':'DISC','amount':-3},{'name':'TAX','amount':-10.0}])==100.0"),
    # --- format-establishing op, then a recall op that must match the format ---
    V2Op("report",
         "Write `report(ledger)` returning a string with one line per entry formatted as '{name}: {amount:.2f}', then a final line 'TOTAL: {t:.2f}' where t is computed by CALLING total(ledger). Lines are joined by '\\n'.",
         ["total"], ["total"], "",
         "assert report([{'name':'a','amount':2}])=='a: 2.00\\nTOTAL: 2.00'",
         "return m.report([{'name':'a','amount':2}])=='a: 2.00\\nTOTAL: 2.00'"),
    V2Op("summary_line",
         "Write `summary_line(ledger)` returning a one-line string 'BASE {b} ADJ {a}' where b is the base total and a is the adjustments total, each formatted with the SAME number of decimal places you used for amounts in report. Reuse base_total and adjustments_total.",
         ["base_total", "adjustments_total"],
         ["base_total", "adjustments_total"],
         "Amounts in report are formatted to 2 decimal places. The two quantities are base_total(ledger) and adjustments_total(ledger).",
         "assert summary_line([{'name':'a','amount':100},{'name':'FEE','amount':-4},{'name':'DISC','amount':-3},{'name':'TAX','amount':-10.0}])=='BASE 100.00 ADJ -17.00'",
         "return m.summary_line([{'name':'a','amount':100},{'name':'FEE','amount':-4},{'name':'DISC','amount':-3},{'name':'TAX','amount':-10.0}])=='BASE 100.00 ADJ -17.00'"),
    # --- light, low-coupling ops for length variety beyond the coupled spine ---
    V2Op("count_entries",
         "Write `count_entries(ledger)` returning the number of entries.",
         [], [], "",
         "assert count_entries([])==0\nassert count_entries([{'name':'a','amount':1}])==1",
         "return m.count_entries([])==0 and m.count_entries([{'name':'a','amount':1}])==1"),
    V2Op("names_list",
         "Write `names_list(ledger)` returning a list of the 'name' fields in order, including duplicates.",
         [], [], "",
         "assert names_list([{'name':'a','amount':1},{'name':'a','amount':2}])==['a','a']",
         "return m.names_list([{'name':'a','amount':1},{'name':'a','amount':2}])==['a','a']"),
    V2Op("positive_entries",
         "Write `positive_entries(ledger)` returning a list of entries whose 'amount' > 0, preserving order.",
         [], [], "",
         "assert positive_entries([{'name':'a','amount':-1},{'name':'b','amount':2}])==[{'name':'b','amount':2}]",
         "return m.positive_entries([{'name':'a','amount':-1},{'name':'b','amount':2}])==[{'name':'b','amount':2}]"),
    V2Op("clear_ledger",
         "Write `clear_ledger(ledger)` that removes all entries in place (leaving the same list object empty) and returns None.",
         [], [], "",
         "l=[{'name':'a','amount':1}]\nclear_ledger(l)\nassert l==[]",
         "l=[{'name':'a','amount':1}]; m.clear_ledger(l); return l==[]"),
]
V2_BY_NAME: dict[str, V2Op] = {op.name: op for op in V2_OP_BANK}
V2_INDEX: dict[str, int] = {op.name: i for i, op in enumerate(V2_OP_BANK)}

# Reference-correct source for EVERY op. Used (a) to seed transitively-called upstream
# in solo calibration / position-controlled runs, and (b) by the mock provider.
V2_REFS: dict[str, str] = {
    "add_entry": "def add_entry(ledger, name, amount):\n    ledger.append({'name': name, 'amount': amount})\n",
    "total": "def total(ledger):\n    return sum(e['amount'] for e in ledger)\n",
    "apply_fee": "def apply_fee(ledger, fee):\n    add_entry(ledger, 'FEE', -fee)\n",
    "apply_discount": "def apply_discount(ledger, disc):\n    add_entry(ledger, 'DISC', -disc)\n",
    "apply_tax": "def apply_tax(ledger, rate):\n    t = total(ledger)\n    add_entry(ledger, 'TAX', -t * rate)\n",
    "fee_total": "def fee_total(ledger):\n    return sum(e['amount'] for e in ledger if e['name'] == 'FEE')\n",
    "discount_total": "def discount_total(ledger):\n    return sum(e['amount'] for e in ledger if e['name'] == 'DISC')\n",
    "tax_total": "def tax_total(ledger):\n    return sum(e['amount'] for e in ledger if e['name'] == 'TAX')\n",
    "adjustments_total": "def adjustments_total(ledger):\n    return fee_total(ledger) + discount_total(ledger) + tax_total(ledger)\n",
    "base_total": "def base_total(ledger):\n    return total(ledger) - adjustments_total(ledger)\n",
    "report": ("def report(ledger):\n"
               "    lines = [f\"{e['name']}: {e['amount']:.2f}\" for e in ledger]\n"
               "    lines.append(f\"TOTAL: {total(ledger):.2f}\")\n"
               "    return \"\\n\".join(lines)\n"),
    "summary_line": "def summary_line(ledger):\n    return f\"BASE {base_total(ledger):.2f} ADJ {adjustments_total(ledger):.2f}\"\n",
    "count_entries": "def count_entries(ledger):\n    return len(ledger)\n",
    "names_list": "def names_list(ledger):\n    return [e['name'] for e in ledger]\n",
    "positive_entries": "def positive_entries(ledger):\n    return [e for e in ledger if e['amount'] > 0]\n",
    "clear_ledger": "def clear_ledger(ledger):\n    ledger.clear()\n",
}


# --------------------------------------------------------------------------- #
# v3 OP BANK — "pricing session" with ACCUMULATING recall load (Brief, 19 Jun).
# v2 was flat / single-op because difficulty sat at fixed positions with filler in
# between, so a longer chain wasn't a harder chain. Here numeric parameters are SET
# and then UPDATED across the chain (unit price 6→9→12, discount 3→5, service 10→7),
# and "integrator" ops must compute with the value CURRENTLY in effect. Each integrator
# is at-ceiling in isolation (calibration supplies the current values via `recall`); in
# the accumulating chain the value lives only in earlier prompts, so the agent must
# track the latest of several evolving parameters. Anchoring on a STALE value (which is
# sitting right there in an earlier turn) yields a graceful, exact-testable wrong answer.
# Longer N => more parameters set, more updates, more distance back => more to track.
# All integer arithmetic (exact). Integrators are pure (no delegation to a getter), so
# the recalled value must be committed, not re-derived.
# --------------------------------------------------------------------------- #
V3_OP_BANK: list[V2Op] = [
    # --- warm-ups (no recall) ---
    V2Op("add_entry",
         "Write `add_entry(ledger, name, amount)` that appends {'name': name, 'amount': amount} to `ledger` in place and returns None.",
         [], [], "",
         "l=[]\nadd_entry(l,'a',5)\nassert l==[{'name':'a','amount':5}]",
         "l=[]; m.add_entry(l,'a',5); return l==[{'name':'a','amount':5}]"),
    V2Op("count_entries",
         "Write `count_entries(ledger)` returning the number of entries.",
         [], [], "",
         "assert count_entries([])==0\nassert count_entries([{'name':'a','amount':1}])==1",
         "return m.count_entries([])==0 and m.count_entries([{'name':'a','amount':1}])==1"),
    # --- establish parameters (value stated in the op's own spec) ---
    V2Op("record_units",
         "The unit price for this session is 6. Write `record_units(ledger, qty)` that appends {'name': 'UNIT', 'amount': 6 * qty} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_units(l,3)\nassert l==[{'name':'UNIT','amount':18}]",
         "l=[]; m.record_units(l,3); return l==[{'name':'UNIT','amount':18}]"),
    V2Op("record_service",
         "The service fee is 10. Write `record_service(ledger)` that appends {'name': 'SVC', 'amount': -10} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_service(l)\nassert l==[{'name':'SVC','amount':-10}]",
         "l=[]; m.record_service(l); return l==[{'name':'SVC','amount':-10}]"),
    V2Op("record_discount",
         "The discount is 3 per unit. Write `record_discount(ledger, units)` that appends {'name': 'DISC', 'amount': -3 * units} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_discount(l,4)\nassert l==[{'name':'DISC','amount':-12}]",
         "l=[]; m.record_discount(l,4); return l==[{'name':'DISC','amount':-12}]"),
    # --- integrator: recall current unit price (6) ---
    V2Op("price_of",
         "Write `price_of(qty)` returning qty multiplied by the current unit price.",
         [], [], "The current unit price is 6.",
         "assert price_of(5)==30\nassert price_of(0)==0",
         "return m.price_of(5)==30 and m.price_of(0)==0"),
    # --- UPDATE unit price 6 -> 9 ---
    V2Op("record_units_now",
         "Update: the unit price is now 9. Write `record_units_now(ledger, qty)` that appends {'name': 'UNIT', 'amount': 9 * qty} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_units_now(l,2)\nassert l==[{'name':'UNIT','amount':18}]",
         "l=[]; m.record_units_now(l,2); return l==[{'name':'UNIT','amount':18}]"),
    # --- integrator: recall CURRENT (updated) unit price (9), not the original 6 ---
    V2Op("current_price_of",
         "Write `current_price_of(qty)` returning qty multiplied by the unit price currently in effect.",
         [], [], "The current unit price is 9.",
         "assert current_price_of(3)==27",
         "return m.current_price_of(3)==27 and m.current_price_of(1)==9"),
    V2Op("record_credit",
         "The membership credit is 8. Write `record_credit(ledger)` that appends {'name': 'CREDIT', 'amount': 8} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_credit(l)\nassert l==[{'name':'CREDIT','amount':8}]",
         "l=[]; m.record_credit(l); return l==[{'name':'CREDIT','amount':8}]"),
    # --- integrator: current unit price (9) AND current discount (3) ---
    V2Op("net_unit_cost",
         "Write `net_unit_cost(units)` returning units times (the current unit price minus the current per-unit discount).",
         [], [], "The current unit price is 9 and the current per-unit discount is 3.",
         "assert net_unit_cost(5)==30",
         "return m.net_unit_cost(5)==30 and m.net_unit_cost(0)==0"),
    # --- UPDATE discount 3 -> 5, service 10 -> 7 ---
    V2Op("record_discount_now",
         "Update: the discount is now 5 per unit. Write `record_discount_now(ledger, units)` that appends {'name': 'DISC', 'amount': -5 * units} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_discount_now(l,2)\nassert l==[{'name':'DISC','amount':-10}]",
         "l=[]; m.record_discount_now(l,2); return l==[{'name':'DISC','amount':-10}]"),
    V2Op("record_service_now",
         "Update: the service fee is now 7. Write `record_service_now(ledger)` that appends {'name': 'SVC', 'amount': -7} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_service_now(l)\nassert l==[{'name':'SVC','amount':-7}]",
         "l=[]; m.record_service_now(l); return l==[{'name':'SVC','amount':-7}]"),
    # --- integrator: current unit price (9) AND current discount (now 5, updated) ---
    V2Op("net_unit_cost_now",
         "Write `net_unit_cost_now(units)` returning units times (the current unit price minus the current per-unit discount).",
         [], [], "The current unit price is 9 and the current per-unit discount is 5.",
         "assert net_unit_cost_now(5)==20",
         "return m.net_unit_cost_now(5)==20 and m.net_unit_cost_now(2)==8"),
    # --- integrator: current service fee (now 7, updated from 10) ---
    V2Op("service_charge",
         "Write `service_charge()` returning the service fee currently in effect, as a positive integer.",
         [], [], "The current service fee is 7.",
         "assert service_charge()==7",
         "return m.service_charge()==7"),
    V2Op("record_handling",
         "The handling fee is 2 per parcel. Write `record_handling(ledger, parcels)` that appends {'name': 'HANDLING', 'amount': -2 * parcels} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_handling(l,3)\nassert l==[{'name':'HANDLING','amount':-6}]",
         "l=[]; m.record_handling(l,3); return l==[{'name':'HANDLING','amount':-6}]"),
    # --- heavy integrator: current price (9), current discount (5), service (7), credit (8) ---
    V2Op("bundle_cost",
         "Write `bundle_cost(units)` returning units*(current unit price - current per-unit discount) + current service fee - membership credit.",
         [], [], "Current unit price is 9, current per-unit discount is 5, current service fee is 7, membership credit is 8.",
         "assert bundle_cost(3)==11",
         "return m.bundle_cost(3)==11 and m.bundle_cost(0)==-1"),
    # --- UPDATE unit price 9 -> 12 ---
    V2Op("record_units_latest",
         "Update: the unit price is now 12. Write `record_units_latest(ledger, qty)` that appends {'name': 'UNIT', 'amount': 12 * qty} to `ledger` and returns None.",
         [], [], "",
         "l=[]\nrecord_units_latest(l,2)\nassert l==[{'name':'UNIT','amount':24}]",
         "l=[]; m.record_units_latest(l,2); return l==[{'name':'UNIT','amount':24}]"),
    # --- heaviest integrator: LATEST price (12), current discount (5), service (7), credit (8) ---
    V2Op("bundle_cost_latest",
         "Write `bundle_cost_latest(units)` returning units*(current unit price - current per-unit discount) + current service fee - membership credit.",
         [], [], "Current unit price is 12, current per-unit discount is 5, current service fee is 7, membership credit is 8.",
         "assert bundle_cost_latest(3)==20",
         "return m.bundle_cost_latest(3)==20 and m.bundle_cost_latest(1)==6"),
]

V3_REFS: dict[str, str] = {
    "add_entry": "def add_entry(ledger, name, amount):\n    ledger.append({'name': name, 'amount': amount})\n",
    "count_entries": "def count_entries(ledger):\n    return len(ledger)\n",
    "record_units": "def record_units(ledger, qty):\n    ledger.append({'name': 'UNIT', 'amount': 6 * qty})\n",
    "record_service": "def record_service(ledger):\n    ledger.append({'name': 'SVC', 'amount': -10})\n",
    "record_discount": "def record_discount(ledger, units):\n    ledger.append({'name': 'DISC', 'amount': -3 * units})\n",
    "price_of": "def price_of(qty):\n    return 6 * qty\n",
    "record_units_now": "def record_units_now(ledger, qty):\n    ledger.append({'name': 'UNIT', 'amount': 9 * qty})\n",
    "current_price_of": "def current_price_of(qty):\n    return 9 * qty\n",
    "record_credit": "def record_credit(ledger):\n    ledger.append({'name': 'CREDIT', 'amount': 8})\n",
    "net_unit_cost": "def net_unit_cost(units):\n    return units * (9 - 3)\n",
    "record_discount_now": "def record_discount_now(ledger, units):\n    ledger.append({'name': 'DISC', 'amount': -5 * units})\n",
    "record_service_now": "def record_service_now(ledger):\n    ledger.append({'name': 'SVC', 'amount': -7})\n",
    "net_unit_cost_now": "def net_unit_cost_now(units):\n    return units * (9 - 5)\n",
    "service_charge": "def service_charge():\n    return 7\n",
    "record_handling": "def record_handling(ledger, parcels):\n    ledger.append({'name': 'HANDLING', 'amount': -2 * parcels})\n",
    "bundle_cost": "def bundle_cost(units):\n    return units * (9 - 5) + 7 - 8\n",
    "record_units_latest": "def record_units_latest(ledger, qty):\n    ledger.append({'name': 'UNIT', 'amount': 12 * qty})\n",
    "bundle_cost_latest": "def bundle_cost_latest(units):\n    return units * (12 - 5) + 7 - 8\n",
}

# --------------------------------------------------------------------------- #
# v4 OP BANK — COMPOSITIONAL DEPTH that accumulates with N (Brief, 19 Jun).
# Evidence from v2/v3: what bends a model is relational composition over many
# abstractions (v2 `base_total`), NOT memory/value-recall (v3 was flat — values are
# re-readable from the agent's own code). So here each "category" op writes a trivial
# magnitude-sum function (`sale_total` = sum of 'SALE' amounts), and its ROLE (charge
# vs credit) is stated only in that op's prompt — never encoded in the function. The
# "running net" aggregates must compose EVERY category-so-far with the correct sign,
# recalling each role. Depth grows down the chain (net_a composes 3, net_e composes 11),
# so a longer chain is a genuinely harder chain. At-ceiling in isolation (calibration
# seeds the category functions + supplies the roles); in-chain a missed category or a
# flipped sign is an exact-testable graceful wrong answer. All integer arithmetic.
# --------------------------------------------------------------------------- #
def _cat(name: str, sentinel: str, role: str) -> V2Op:
    return V2Op(
        f"{name}_total",
        f"{sentinel} entries are {role}. Write `{name}_total(ledger)` returning the sum of "
        f"'amount' for entries named '{sentinel}' (0 if none).",
        [], [], "",
        f"assert {name}_total([{{'name':'{sentinel}','amount':5}},{{'name':'X','amount':9}}])==5",
        f"return m.{name}_total([{{'name':'{sentinel}','amount':5}},{{'name':'X','amount':9}}])==5 "
        f"and m.{name}_total([])==0")


# full-category test ledger reused by every aggregate's held-out test
_L4 = ("L=[{'name':n,'amount':a} for n,a in [('SALE',100),('SHIP',10),('REFUND',20),"
       "('TAX',8),('COUPON',7),('FEE',5),('REWARD',4),('SURCHARGE',3),('ADJUST',2),"
       "('HANDLING',6),('REBATE',1)]]")


def _agg(name: str, charges: list[str], credits: list[str], expected: int) -> V2Op:
    calls = [f"{c}_total" for c in charges] + [f"{c}_total" for c in credits]
    roles = ("Charges so far: " + ", ".join(c.upper() for c in charges)
             + ". Credits so far: " + ", ".join(c.upper() for c in credits) + ".")
    return V2Op(
        name,
        f"Write `{name}(ledger)` returning (sum of all charge-category totals) minus (sum of all "
        f"credit-category totals) across every category introduced so far, by calling each "
        f"category's `_total` function with + if it is a charge and - if it is a credit.",
        calls, calls, roles,
        f"assert {name}([{{'name':'SALE','amount':5}}])==5",
        f"{_L4}\nreturn m.{name}(L)=={expected}")


V4_OP_BANK: list[V2Op] = [
    V2Op("add_entry",
         "Write `add_entry(ledger, name, amount)` that appends {'name': name, 'amount': amount} to `ledger` in place and returns None.",
         [], [], "",
         "l=[]\nadd_entry(l,'a',5)\nassert l==[{'name':'a','amount':5}]",
         "l=[]; m.add_entry(l,'a',5); return l==[{'name':'a','amount':5}]"),
    _cat("sale", "SALE", "charges"),
    _cat("ship", "SHIP", "charges"),
    _cat("refund", "REFUND", "credits"),
    _agg("net_a", ["sale", "ship"], ["refund"], 90),
    _cat("tax", "TAX", "charges"),
    _cat("coupon", "COUPON", "credits"),
    _agg("net_b", ["sale", "ship", "tax"], ["refund", "coupon"], 91),
    _cat("fee", "FEE", "charges"),
    _cat("reward", "REWARD", "credits"),
    _agg("net_c", ["sale", "ship", "tax", "fee"], ["refund", "coupon", "reward"], 92),
    _cat("surcharge", "SURCHARGE", "charges"),
    _cat("adjust", "ADJUST", "credits"),
    _agg("net_d", ["sale", "ship", "tax", "fee", "surcharge"],
         ["refund", "coupon", "reward", "adjust"], 93),
    _cat("handling", "HANDLING", "charges"),
    _cat("rebate", "REBATE", "credits"),
    _agg("net_e", ["sale", "ship", "tax", "fee", "surcharge", "handling"],
         ["refund", "coupon", "reward", "adjust", "rebate"], 98),
]

_CAT_REFS = {
    "sale": "SALE", "ship": "SHIP", "refund": "REFUND", "tax": "TAX", "coupon": "COUPON",
    "fee": "FEE", "reward": "REWARD", "surcharge": "SURCHARGE", "adjust": "ADJUST",
    "handling": "HANDLING", "rebate": "REBATE",
}
V4_REFS: dict[str, str] = {
    "add_entry": "def add_entry(ledger, name, amount):\n    ledger.append({'name': name, 'amount': amount})\n",
    **{f"{n}_total": f"def {n}_total(ledger):\n    return sum(e['amount'] for e in ledger if e['name'] == '{s}')\n"
       for n, s in _CAT_REFS.items()},
    "net_a": "def net_a(ledger):\n    return sale_total(ledger) + ship_total(ledger) - refund_total(ledger)\n",
    "net_b": "def net_b(ledger):\n    return sale_total(ledger) + ship_total(ledger) + tax_total(ledger) - refund_total(ledger) - coupon_total(ledger)\n",
    "net_c": "def net_c(ledger):\n    return sale_total(ledger) + ship_total(ledger) + tax_total(ledger) + fee_total(ledger) - refund_total(ledger) - coupon_total(ledger) - reward_total(ledger)\n",
    "net_d": "def net_d(ledger):\n    return sale_total(ledger) + ship_total(ledger) + tax_total(ledger) + fee_total(ledger) + surcharge_total(ledger) - refund_total(ledger) - coupon_total(ledger) - reward_total(ledger) - adjust_total(ledger)\n",
    "net_e": "def net_e(ledger):\n    return sale_total(ledger) + ship_total(ledger) + tax_total(ledger) + fee_total(ledger) + surcharge_total(ledger) + handling_total(ledger) - refund_total(ledger) - coupon_total(ledger) - reward_total(ledger) - adjust_total(ledger) - rebate_total(ledger)\n",
}

# Bank registry — select with --bank. Each maps name -> (ops, reference impls).
BANKS: dict[str, list[V2Op]] = {"v2": V2_OP_BANK, "v3": V3_OP_BANK, "v4": V4_OP_BANK}
REFS_BY_BANK: dict[str, dict[str, str]] = {"v2": V2_REFS, "v3": V3_REFS, "v4": V4_REFS}


def _called_closure(op_name: str, by_name: dict[str, V2Op], acc: set[str] | None = None) -> set[str]:
    """Transitive set of functions an op invokes at runtime (excludes the op itself)."""
    acc = acc if acc is not None else set()
    for callee in by_name[op_name].calls:
        if callee not in acc:
            acc.add(callee)
            _called_closure(callee, by_name, acc)
    return acc


def grade_step(modpath: Path, op: V2Op, timeout: int = 8) -> bool:
    """v1 grader (reads op.held_out / op.name); V2Op is structurally an Op for its use."""
    return bool(_grade_step(str(modpath), cast(Any, op), timeout))


def run_visible(modpath: Path, op: V2Op, timeout: int = 8) -> str:
    return str(_run_visible(str(modpath), cast(Any, op), timeout))


def module_import_error(modpath: Path, timeout: int = 8) -> str:
    """Return '' if ledger.py imports cleanly, else the repr of the import-time error.
    A non-empty result means the agent corrupted the shared module — a catastrophic
    failure distinct from a single step computing the wrong value (Brief: severity)."""
    script = (
        "import importlib.util,sys\n"
        f"spec=importlib.util.spec_from_file_location('ledger',{str(modpath)!r})\n"
        "m=importlib.util.module_from_spec(spec)\n"
        "try:\n    spec.loader.exec_module(m)\n    print('IMPORT_OK')\n"
        "except Exception as e:\n    print('IMPORT_FAIL:'+repr(e))\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TimeoutExpired"
    so = out.stdout or ""
    if "IMPORT_OK" in so:
        return ""
    marker = "IMPORT_FAIL:"
    return so[so.index(marker) + len(marker):].strip() if marker in so else "unknown import failure"


def _clean_code(raw: str) -> str:
    code = raw.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.rstrip()


def try_append(modpath: Path, raw_code: str) -> str:
    """Transactional append: only commit the new function if the WHOLE module still imports.
    Returns '' on success, or the import/parse error (file left UNCHANGED) on rejection.
    This stops a single malformed append from silently corrupting the chain, and turns
    'broke the file' into a recoverable, observable event (agent retries; we count it)."""
    code = _clean_code(raw_code)
    current = modpath.read_text() if modpath.exists() else ""
    candidate = current + "\n" + code + "\n"
    tmp = modpath.with_name("_candidate.py")
    tmp.write_text(candidate)
    err = module_import_error(tmp)
    tmp.unlink()
    if err:
        return err
    modpath.write_text(candidate)
    return ""


SYSTEM = (
    "You are a precise Python coding agent incrementally building ONE module, ledger.py, "
    "across a sequence of tasks in a single session. A ledger is a list of dicts shaped like "
    "{'name': str, 'amount': float}. Functions you wrote in earlier tasks remain in the file and "
    "you may call them; rely on the conventions and names you established earlier and keep them "
    "consistent. For each task, use append_function to add ONLY the new function (do not redefine "
    "functions that already exist). You may call run_tests to check the visible example, then call "
    "done(). Call abstain(reason) only if you genuinely cannot complete the task correctly."
)


# --------------------------------------------------------------------------- #
# LLM adapter: anthropic (with rolling prefix caching) / openai / mock.
# Normalized step() returns: (text, [(tool_name, args, call_id)], usage, stop_reason)
# and appends the assistant turn to `history` in place.
# --------------------------------------------------------------------------- #
class LLMv2:
    def __init__(self, provider: str, model: str, temperature: float,
                 abstain_enabled: bool, cache: bool = False, bug: str | None = None,
                 refs: dict[str, str] | None = None) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.cache = cache
        self.bug = bug
        self.refs = refs if refs is not None else V2_REFS  # reference impls for the mock
        self.tools = TOOLS if abstain_enabled else [t for t in TOOLS if t["name"] != "abstain"]
        self.client: Any = None
        self._retry_exc: tuple[type[BaseException], ...] = ()
        if provider == "anthropic":
            import anthropic
            # SDK auto-retries transient errors with backoff; bump from the default 2.
            self.client = anthropic.Anthropic(max_retries=6)
            # Outer belt-and-suspenders retry for sustained 429/5xx/529 (overload) spikes.
            self._retry_exc = (anthropic.APIStatusError, anthropic.APIConnectionError)
            if cache:
                self.system_param: Any = [{"type": "text", "text": SYSTEM,
                                           "cache_control": {"type": "ephemeral"}}]
            else:
                self.system_param = SYSTEM
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI(max_retries=6)
            self.otools = to_openai_tools(self.tools)
        elif provider == "mock":
            pass
        else:
            raise ValueError("provider must be 'anthropic', 'openai' or 'mock'")

    # -- mock: deterministic, stateless (decides from history) -------------- #
    def _mock_step(self, history: list[dict[str, Any]]
                   ) -> tuple[str, list[tuple[str, dict[str, Any], str]], dict[str, int], str]:
        tasks = [m for m in history if m.get("role") == "user"
                 and isinstance(m.get("content"), list)
                 and any(isinstance(b, dict) and b.get("type") == "text"
                         and str(b.get("text", "")).startswith("TASK[op=") for b in m["content"])]
        appends = sum(1 for m in history if m.get("role") == "assistant"
                      and isinstance(m.get("content"), list)
                      and any(isinstance(b, dict) and b.get("type") == "tool_use"
                              and b.get("name") == "append_function" for b in m["content"]))
        usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
        cid = "mock_%d" % len(history)
        calls: list[tuple[str, dict[str, Any], str]]
        if appends < len(tasks):  # still owe the append for the current op
            text = str(tasks[-1]["content"][0]["text"])
            mo = re.search(r"TASK\[op=(\w+)\]", text)
            name = mo.group(1) if mo else ""
            if self.bug == name:
                code = "def %s(*args, **kwargs):\n    return None\n" % name  # deliberately wrong
            else:
                code = self.refs.get(name, "def _missing():\n    return None\n")
            calls = [("append_function", {"code": code}, cid)]
        else:
            calls = [("done", {}, cid)]
        history.append({"role": "assistant",
                        "content": [{"type": "tool_use", "id": cid,
                                     "name": calls[0][0], "input": calls[0][1]}]})
        return "", calls, usage, "tool_use"

    def _anthropic_create(self, history: list[dict[str, Any]]) -> Any:
        """messages.create with an outer exponential backoff on top of the SDK's own retries,
        so a sustained overload (429/529) spike during a long batch does not kill the whole run."""
        delay = 4.0
        for attempt in range(5):
            try:
                return self.client.messages.create(
                    model=self.model, max_tokens=1500, temperature=self.temperature,
                    system=self.system_param, tools=self.tools, messages=history)
            except self._retry_exc as e:
                status = getattr(e, "status_code", None)
                retriable = status is None or status == 429 or status >= 500
                if not retriable or attempt == 4:
                    raise
                print(f"  [retry] {type(e).__name__} status={status}; "
                      f"sleeping {delay:.0f}s (attempt {attempt + 1}/5)", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def step(self, history: list[dict[str, Any]]
             ) -> tuple[str, list[tuple[str, dict[str, Any], str]], dict[str, int], str]:
        if self.provider == "mock":
            return self._mock_step(history)
        if self.provider == "anthropic":
            resp = self._anthropic_create(history)
            text = ""
            calls: list[tuple[str, dict[str, Any], str]] = []
            for b in resp.content:
                if b.type == "text":
                    text += b.text
                elif b.type == "tool_use":
                    calls.append((b.name, b.input or {}, b.id))
            history.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
            u = resp.usage
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            usage = {"in": u.input_tokens + cr + cw, "out": u.output_tokens,
                     "cache_read": cr, "cache_write": cw}
            return text, calls, usage, resp.stop_reason
        # openai
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=1500,
            tools=self.otools, messages=history)
        msg = resp.choices[0].message
        ocalls: list[tuple[str, dict[str, Any], str]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    a = {}
                ocalls.append((tc.function.name, a, tc.id))
        history.append(msg.model_dump(exclude_none=True))
        u = resp.usage
        usage = {"in": u.prompt_tokens, "out": u.completion_tokens, "cache_read": 0, "cache_write": 0}
        return (msg.content or ""), ocalls, usage, resp.choices[0].finish_reason

    def tool_result_msg(self, call_id: str, content: str) -> dict[str, Any]:
        if self.provider == "openai":
            return {"role": "tool", "tool_call_id": call_id, "content": content}
        return {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": content}]}


def _set_cache_breakpoint(history: list[dict[str, Any]]) -> None:
    """Move the single rolling cache breakpoint to the last block of the last message.
    Strips stale breakpoints first so the growing prefix caches incrementally (Design §14)."""
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict):
                    blk.pop("cache_control", None)
    if history:
        last = history[-1].get("content")
        if isinstance(last, list) and last and isinstance(last[-1], dict):
            last[-1]["cache_control"] = {"type": "ephemeral"}


def _context_block(op: V2Op, available: list[str]) -> str:
    lines = []
    if op.recall:
        lines.append("- " + op.recall)
    lines.append("- Functions already defined that you may call: "
                 + (", ".join(available) or "none yet"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# One sub-task: the agent works `op` to done/abstain/cap, appending to `history`.
# `task_text` is built by the caller (with/without recall+signature context).
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    terminal: str                 # done | abstain | maxsteps
    reason: str                   # abstain reason ('' otherwise)
    tok_in: int
    tok_out: int
    cache_read: int
    cache_write: int
    rejected: int                 # # of append attempts rejected for not parsing/importing


def run_step(llm: LLMv2, modpath: Path, op: V2Op, max_steps: int,
             history: list[dict[str, Any]], task_text: str, cache_mark: bool) -> StepResult:
    if not modpath.exists():
        modpath.write_text("")
    if llm.provider == "openai":
        history.append({"role": "user", "content": task_text})
    else:
        history.append({"role": "user", "content": [{"type": "text", "text": task_text}]})
        if cache_mark:
            _set_cache_breakpoint(history)

    ti = to = cr = cw = rejected = 0
    reason = ""
    for _ in range(max_steps):
        _text, calls, usage, _stop = llm.step(history)
        ti += usage["in"]
        to += usage["out"]
        cr += usage.get("cache_read", 0)
        cw += usage.get("cache_write", 0)
        if not calls:
            nudge = "Use a tool, or call done() when finished."
            if llm.provider == "openai":
                history.append({"role": "user", "content": nudge})
            else:
                history.append({"role": "user", "content": [{"type": "text", "text": nudge}]})
            continue
        terminal: str | None = None
        for (name, args, cid) in calls:
            if name == "append_function":
                err = try_append(modpath, str(args.get("code", "")))
                if err:
                    rejected += 1
                    result = ("NOT appended: that code did not parse or broke the module "
                              f"({err}). The file is UNCHANGED — fix it and append again.")
                else:
                    result = "appended"
            elif name == "run_tests":
                result = run_visible(modpath, op)
            elif name == "done":
                result = "ok"
                terminal = "done"
            elif name == "abstain":
                result = "ok"
                terminal = "abstain"
                reason = str(args.get("reason", ""))
            else:
                result = "unknown tool"
            history.append(llm.tool_result_msg(cid, result))
        if terminal:
            return StepResult(terminal, reason, ti, to, cr, cw, rejected)
    return StepResult("maxsteps", "", ti, to, cr, cw, rejected)


# --------------------------------------------------------------------------- #
# A chain of length N over the same module. `context_mode`:
#   accumulate -> ONE history across all steps; no signature/recall crutch re-injected.
#   reset      -> fresh history per step; recall facts + current signatures re-handed.
# --------------------------------------------------------------------------- #
@dataclass
class RunRecord:
    N: int
    context_mode: str
    outcome: str                  # success | wrong | corrupted | abstained | maxsteps
    per_step_pass: list[bool]
    terminals: list[str]
    first_fail: int               # -1 if none (or if module didn't import: not meaningful)
    propagated: bool
    abstained_at: int             # -1 if none
    imported: bool                # did the final module import? False => catastrophic corruption
    import_error: str             # repr of the import-time error ('' if clean)
    abstain_reason: str           # the agent's stated reason if it abstained ('' otherwise)
    rejected_appends: int         # appends rejected for not parsing/importing (recovery/severity)
    tok_in: int
    tok_out: int
    cache_read: int
    cache_write: int
    module_src: str               # final ledger.py source, saved only on non-success (else '')


def run_chain(llm: LLMv2, ops: list[V2Op], max_steps: int,
              context_mode: str, cache: bool) -> RunRecord:
    with tempfile.TemporaryDirectory() as d:
        modpath = Path(d) / "ledger.py"
        modpath.write_text("")
        tin = tout = cread = cwrite = rejected = 0
        terminals: list[str] = []
        abstained_at = -1
        abstain_reason = ""

        if context_mode == "accumulate":
            history: list[dict[str, Any]] = (
                [{"role": "system", "content": SYSTEM}] if llm.provider == "openai" else [])
            for i, op in enumerate(ops):
                task = (f"TASK[op={op.name}] {op.spec}\n"
                        "Add only this new function with append_function, then call done().")
                sr = run_step(
                    llm, modpath, op, max_steps, history, task,
                    cache_mark=(cache and llm.provider == "anthropic"))
                terminals.append(sr.terminal)
                tin += sr.tok_in
                tout += sr.tok_out
                cread += sr.cache_read
                cwrite += sr.cache_write
                rejected += sr.rejected
                if sr.terminal == "abstain":
                    abstained_at = i
                    abstain_reason = sr.reason
                    break
        else:  # reset
            for i, op in enumerate(ops):
                history = ([{"role": "system", "content": SYSTEM}]
                           if llm.provider == "openai" else [])
                ctx = _context_block(op, sigs(modpath))
                task = (f"TASK[op={op.name}] {op.spec}\n\nContext you can rely on:\n{ctx}\n"
                        "Add only this new function with append_function "
                        "(do not redefine existing functions), then call done().")
                sr = run_step(
                    llm, modpath, op, max_steps, history, task, cache_mark=False)
                terminals.append(sr.terminal)
                tin += sr.tok_in
                tout += sr.tok_out
                cread += sr.cache_read
                cwrite += sr.cache_write
                rejected += sr.rejected
                if sr.terminal == "abstain":
                    abstained_at = i
                    abstain_reason = sr.reason
                    break

        graded = ops if abstained_at < 0 else ops[:abstained_at]
        import_error = module_import_error(modpath)
        imported = (import_error == "")
        per_step = [grade_step(modpath, op) for op in graded]

        if not imported:
            # Catastrophic corruption: per-step grades are all-fail artefacts of the import
            # failure, so first_fail / propagated are not meaningful here.
            first_fail = -1
            propagated = False
        else:
            first_fail = next((i for i, ok in enumerate(per_step) if not ok), -1)
            name_to_idx = {op.name: i for i, op in enumerate(graded)}
            propagated = False
            for i, op in enumerate(graded):
                if not per_step[i]:
                    for dep in op.deps:
                        j = name_to_idx.get(dep)
                        if j is not None and not per_step[j]:
                            propagated = True

        if abstained_at >= 0:
            outcome = "abstained"
        elif not imported:
            outcome = "corrupted"
        elif per_step and all(per_step):
            outcome = "success"
        elif "maxsteps" in terminals:
            outcome = "maxsteps"
        else:
            outcome = "wrong"
        module_src = modpath.read_text() if outcome != "success" else ""
        return RunRecord(len(ops), context_mode, outcome, per_step, terminals,
                         first_fail, propagated, abstained_at, imported, import_error,
                         abstain_reason, rejected, tin, tout, cread, cwrite, module_src)


# --------------------------------------------------------------------------- #
# Light calibration: each op SOLO, abstain off, transitively-called upstream seeded,
# recall facts + signatures provided -> per-op p̂ for the ∏p̂ null (Design §6, §8).
# --------------------------------------------------------------------------- #
def _calib_trial(llm: LLMv2, op: V2Op, max_steps: int,
                 refs: dict[str, str], by_name: dict[str, V2Op]) -> bool:
    with tempfile.TemporaryDirectory() as d:
        mp = Path(d) / "ledger.py"
        # seed transitively-called upstream (order within the file is irrelevant for defs)
        closure = _called_closure(op.name, by_name)
        mp.write_text("".join(refs[n] for n in closure))
        history: list[dict[str, Any]] = (
            [{"role": "system", "content": SYSTEM}] if llm.provider == "openai" else [])
        ctx = _context_block(op, sigs(mp))
        task = (f"TASK[op={op.name}] {op.spec}\n\nContext you can rely on:\n{ctx}\n"
                "Add only this new function with append_function "
                "(do not redefine existing functions), then call done().")
        run_step(llm, mp, op, max_steps, history, task, cache_mark=False)
        # guard: if the agent clobbered a seeded upstream, re-inject it before grading
        content = mp.read_text()
        missing = "".join(refs[n] for n in closure if f"def {n}(" not in content)
        if missing:
            mp.write_text(missing + content)
        return grade_step(mp, op)


def calibrate(provider: str, model: str, temperature: float, n: int, max_steps: int,
              ops: list[V2Op], cache: bool, out: Any, workers: int,
              bug: str | None, refs: dict[str, str], by_name: dict[str, V2Op]) -> dict[str, float]:
    llm = LLMv2(provider, model, temperature, abstain_enabled=False, cache=cache, bug=bug, refs=refs)
    phat: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for op in ops:
            futs = [ex.submit(_calib_trial, llm, op, max_steps, refs, by_name) for _ in range(n)]
            passes = sum(1 for fu in futs if fu.result())
            phat[op.name] = passes / n
            print(f"  calib {op.name:18s} p̂={phat[op.name]:.3f}")
            if out is not None:
                out.write(json.dumps({"phase": "calib", "op": op.name,
                                      "phat": phat[op.name], "n": n}) + "\n")
                out.flush()
    return phat


# --------------------------------------------------------------------------- #
def main() -> None:
    # Windows consoles default to cp1252 and can't encode the ∏/p̂ glyphs we print.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    load_dotenv()
    ap = argparse.ArgumentParser(description="v2 accumulating-conversation reliability probe")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "mock"])
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="cheap-but-capable probe model id")
    ap.add_argument("--context-mode", default="accumulate", choices=["accumulate", "reset"],
                    help="accumulate = one growing conversation (the horizon test); "
                         "reset = fresh context per step (the A/B control)")
    ap.add_argument("--Ns", default="4,6,8,10,12,14,16")
    ap.add_argument("--runs", type=int, default=10, help="chain runs per N")
    ap.add_argument("--calib", type=int, default=20, help="solo runs per op for p̂ (0 = skip)")
    ap.add_argument("--max-steps", type=int, default=8, help="tool-loop cap per sub-task")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--cache", action="store_true",
                    help="Anthropic rolling prefix caching (recommended for accumulate)")
    ap.add_argument("--workers", type=int, default=5, help="concurrent API calls")
    ap.add_argument("--mock", action="store_true", help="use the offline deterministic mock model")
    ap.add_argument("--mock-bug", default=None,
                    help="(mock only) emit a deliberately-wrong impl for this op name")
    ap.add_argument("--exclude-ops", default="",
                    help="comma-separated op names to drop from the bank (per-model calibration "
                         "gate: exclude ops a model fails in isolation, so chain failures are "
                         "reliability not capability — Design §6)")
    ap.add_argument("--bank", default="v4", choices=sorted(BANKS),
                    help="op bank: v2 = coupled-recall (flat/single-op); "
                         "v3 = evolving-parameter recall (flat); "
                         "v4 = accumulating compositional depth (default)")
    args = ap.parse_args()

    provider = "mock" if args.mock else args.provider
    bug = args.mock_bug if provider == "mock" else None
    refs = REFS_BY_BANK[args.bank]
    Ns = [int(x) for x in args.Ns.split(",")]
    maxN = max(Ns)
    excluded = {s.strip() for s in args.exclude_ops.split(",") if s.strip()}
    base = [op for op in BANKS[args.bank] if op.name not in excluded]
    by_name = {op.name: op for op in BANKS[args.bank]}
    print(f"Bank: {args.bank} ({len(base)} ops after exclusions)")
    if excluded:
        print(f"Excluding {len(excluded)} op(s) (not at-ceiling for this model): {sorted(excluded)}")
    ops_seq = [base[i % len(base)] for i in range(maxN)]
    if maxN > len(base):
        print(f"NOTE: N={maxN} exceeds distinct ops ({len(base)}); ops repeat beyond that.")
    if args.context_mode == "accumulate" and provider == "anthropic" and not args.cache:
        print("WARNING: accumulate mode re-sends the growing prefix each call; "
              "without --cache the input-token cost grows ~quadratically in N (Design §10/§14).")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"results_{stamp}.jsonl")

    phat: dict[str, float] = {}
    if args.calib > 0:
        print(f"== Calibration (solo, abstain off, upstream seeded, recall provided), n={args.calib} ==")
        with Path(f"calib_{stamp}.jsonl").open("w") as cf:
            phat = calibrate(provider, args.model, args.temperature, args.calib,
                             args.max_steps, base, args.cache, cf, args.workers, bug,
                             refs, by_name)
        print(f"Calibration records -> calib_{stamp}.jsonl")
        if args.calib_only:
            print("\n--calib-only set; stopping after calibration.")
            return

    print(f"\n== Chains [{args.context_mode}] (abstain on), runs/N={args.runs} ==")
    llm = LLMv2(provider, args.model, args.temperature,
                abstain_enabled=True, cache=args.cache, bug=bug, refs=refs)
    summary: dict[int, dict[str, float]] = {}
    with out_path.open("w") as f:
        for N in Ns:
            ops = ops_seq[:N]
            recs: list[RunRecord] = []
            # Accumulate runs are long, sequential conversations; mock is single-threaded-safe too.
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_chain, llm, ops, args.max_steps, args.context_mode, args.cache)
                        for _ in range(args.runs)]
                for k, fut in enumerate(futs, 1):
                    rec = fut.result()
                    recs.append(rec)
                    f.write(json.dumps(asdict(rec)) + "\n")
                    f.flush()
                    print(f"  N={N:2d} {k:2d}/{args.runs} ({rec.outcome})        ", end="\r")
            n_rec = len(recs)
            succ = sum(1 for x in recs if x.outcome == "success") / n_rec
            abst = sum(1 for x in recs if x.outcome == "abstained") / n_rec
            wrong = sum(1 for x in recs if x.outcome == "wrong") / n_rec
            corr = sum(1 for x in recs if x.outcome == "corrupted") / n_rec
            mx = sum(1 for x in recs if x.outcome == "maxsteps") / n_rec
            rej = sum(x.rejected_appends for x in recs) / n_rec
            toks = sum(x.tok_in + x.tok_out for x in recs) / n_rec
            cread = sum(x.cache_read for x in recs) / n_rec
            tin = sum(x.tok_in for x in recs) / n_rec
            null = 1.0
            for op in ops:
                null *= phat.get(op.name, float("nan"))
            summary[N] = dict(success=succ, abstain=abst, wrong=wrong, corrupted=corr,
                              maxsteps=mx, rejected=rej, mean_tokens=toks, mean_in=tin,
                              cache_read=cread, null=null)
            print(" " * 48, end="\r")
            print(f"  N={N:2d}  success={succ:.2f}  Πp̂={null:.2f}  abstain={abst:.2f}  "
                  f"wrong={wrong:.2f}  corrupted={corr:.2f}  maxsteps={mx:.2f}  "
                  f"rej_appends={rej:.1f}  ~{toks:.0f} tok/run (cache_read {cread:.0f})")

    print(f"\nPer-run records -> {out_path.resolve()}")
    print("\n== Summary (apply the §11 decision rules) ==")
    print(f"{'N':>3} {'success':>8} {'Πp̂(null)':>10} {'dev':>7} {'abstain':>8} "
          f"{'wrong':>7} {'corrupt':>8} {'tok/run':>8} {'cache%':>7}")
    for N in Ns:
        s = summary[N]
        dev = (s["success"] - s["null"]) if s["null"] == s["null"] else float("nan")
        cache_pct = (s["cache_read"] / s["mean_in"] * 100) if s["mean_in"] else 0.0
        print(f"{N:>3} {s['success']:>8.2f} {s['null']:>10.2f} {dev:>+7.2f} "
              f"{s['abstain']:>8.2f} {s['wrong']:>7.2f} {s['corrupted']:>8.2f} "
              f"{s['mean_tokens']:>8.0f} {cache_pct:>6.0f}%")
    print(textwrap.dedent("""
        Reading it (Design §8/§11):
          • success ABOVE Πp̂  -> error-recovery (agent absorbs slips)
          • success TRACKS Πp̂ -> independent compounding (valid finding, proceed)
          • success BELOW Πp̂  -> sub-null failure. SEPARATE the mechanism via the records:
              - corrupted (module won't import)      = catastrophic; agent broke the shared file
              - wrong w/ first_fail at a recall op    = graceful slip / context-degradation (the designed signal)
              - abstained                             = agent flagged it (severity: graceful)
          • FLAT (no decay by max N) -> steps still too easy / coupling too weak -> revise (NOT a finding)
          • Compare accumulate vs reset on identical ops: accumulate bends + reset flat
            => accumulated context is the cause, not step hardness (the v2 A/B; Design §15).
    """))


if __name__ == "__main__":
    main()
