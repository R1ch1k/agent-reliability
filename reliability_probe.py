#!/usr/bin/env python3
"""
reliability_probe.py — Feasibility probe for the Agent Reliability Measurement project.

WHAT THIS DOES
--------------
Runs an LLM agent through COUPLED chains of trivial coding sub-tasks ("ops") of
increasing horizon length N, and measures how reliably it completes the whole chain.
This is the cheap, throwaway feasibility check described in
`Design - step library and feasibility probe.md` (§10). It answers:
  (1) does chain success decay with N, and where;
  (2) how does the empirical curve compare to the independent-failure null  Πp̂ᵢ;
  (3) real tokens-per-run at each N (to firm the main-study budget).

It is NOT the main study. No statistical claims; small samples; one model.

SCAFFOLD (locked design §5): native tool-calling loop with tools
  write_file / read_file / run_tests / done / abstain  + a max-step cap.
Abstain is ENABLED in chain runs, DISABLED in calibration (§6).

SAFETY NOTE: this executes model-generated Python in a subprocess with a timeout,
in a temp dir. The tasks are trivial, but generated code is still untrusted — for
the full study consider a container. Do not run on a machine with sensitive data
without sandboxing.

USAGE
-----
  pip install anthropic            # or: pip install openai
  export ANTHROPIC_API_KEY=sk-...  # or OPENAI_API_KEY=...
  python reliability_probe.py --provider anthropic --model claude-sonnet-4-6 \
         --runs 20 --calib 20

Outputs: results_<timestamp>.jsonl (per-run records) + a printed summary table.
"""

import argparse, json, os, re, subprocess, sys, tempfile, time, textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ----------------------------------------------------------------------------- #
# OP BANK — each op is a trivial function added to a growing ledger.py module.
# A ledger is a list of dicts: {'name': str, 'amount': float}.
# `deps` lists ops whose CORRECTNESS this op relies on (the coupling surface).
# `held_out` is the scoring test: takes the imported module, returns True/False.
# `visible` is a short example shown to the agent + run by run_tests().
# ----------------------------------------------------------------------------- #
@dataclass
class Op:
    name: str
    spec: str                       # natural-language task shown to the agent
    deps: list                      # names of ops this one depends on
    visible: str                    # example/check shown to agent & run by run_tests
    held_out: str                   # python expression body: defines check(m)->bool

OP_BANK = [
    Op("add_entry",
       "Write `add_entry(ledger, name, amount)` that appends {'name': name, 'amount': amount} to the list `ledger` in place and returns None.",
       [],
       "l=[]\nadd_entry(l,'a',5)\nassert l==[{'name':'a','amount':5}]",
       "l=[]; m.add_entry(l,'a',5); return l==[{'name':'a','amount':5}]"),
    Op("total",
       "Write `total(ledger)` returning the sum of all 'amount' values. Empty ledger returns 0.0.",
       [],
       "assert total([])==0.0\nassert total([{'name':'a','amount':2},{'name':'b','amount':3}])==5",
       "return m.total([])==0.0 and m.total([{'name':'a','amount':2},{'name':'b','amount':3}])==5"),
    Op("balance_for",
       "Write `balance_for(ledger, name)` returning the sum of 'amount' for entries whose 'name' equals name. No match returns 0.0.",
       [],
       "assert balance_for([{'name':'a','amount':2},{'name':'a','amount':3}],'a')==5\nassert balance_for([],'x')==0.0",
       "return m.balance_for([{'name':'a','amount':2},{'name':'a','amount':3}],'a')==5 and m.balance_for([],'x')==0.0"),
    Op("apply_fee",
       "Write `apply_fee(ledger, fee)` that records a fee by CALLING add_entry with name 'FEE' and amount -fee. Returns None.",
       ["add_entry"],
       "l=[]\napply_fee(l,4)\nassert l==[{'name':'FEE','amount':-4}]",
       "l=[]; m.apply_fee(l,4); return l==[{'name':'FEE','amount':-4}]"),
    Op("report",
       "Write `report(ledger)` returning a string: one line per entry as '{name}: {amount:.2f}', then a final line 'TOTAL: {t:.2f}' where t is computed by CALLING total(ledger). Lines joined by '\\n'.",
       ["total"],
       "assert report([{'name':'a','amount':2}])=='a: 2.00\\nTOTAL: 2.00'",
       "return m.report([{'name':'a','amount':2}])=='a: 2.00\\nTOTAL: 2.00'"),
    Op("remove_last",
       "Write `remove_last(ledger)` that removes and returns the last entry, or returns None if empty.",
       [],
       "assert remove_last([])is None\nl=[{'name':'a','amount':1}]\nassert remove_last(l)=={'name':'a','amount':1} and l==[]",
       "return m.remove_last([]) is None"),
    Op("count_entries",
       "Write `count_entries(ledger)` returning the number of entries.",
       [],
       "assert count_entries([])==0\nassert count_entries([{'name':'a','amount':1}])==1",
       "return m.count_entries([])==0 and m.count_entries([{'name':'a','amount':1}])==1"),
    Op("entries_for",
       "Write `entries_for(ledger, name)` returning a list of entries whose 'name' equals name, preserving order.",
       [],
       "assert entries_for([{'name':'a','amount':1},{'name':'b','amount':2}],'a')==[{'name':'a','amount':1}]",
       "return m.entries_for([{'name':'a','amount':1},{'name':'b','amount':2}],'a')==[{'name':'a','amount':1}]"),
    Op("max_entry",
       "Write `max_entry(ledger)` returning the entry with the largest 'amount', or None if empty.",
       [],
       "assert max_entry([{'name':'a','amount':1},{'name':'b','amount':9}])=={'name':'b','amount':9}",
       "return m.max_entry([{'name':'a','amount':1},{'name':'b','amount':9}])=={'name':'b','amount':9} and m.max_entry([]) is None"),
    Op("running_balance",
       "Write `running_balance(ledger)` returning a list of cumulative sums of 'amount' (same length as ledger).",
       [],
       "assert running_balance([{'name':'a','amount':2},{'name':'b','amount':3}])==[2,5]",
       "return m.running_balance([{'name':'a','amount':2},{'name':'b','amount':3}])==[2,5] and m.running_balance([])==[]"),
    Op("total_fees",
       "Write `total_fees(ledger)` returning the sum of 'amount' for entries with name 'FEE' (typically negative). No fees returns 0.0.",
       [],
       "assert total_fees([{'name':'FEE','amount':-4},{'name':'a','amount':2}])==-4",
       "return m.total_fees([{'name':'FEE','amount':-4},{'name':'a','amount':2}])==-4 and m.total_fees([])==0.0"),
    Op("average_amount",
       "Write `average_amount(ledger)` returning the mean of 'amount' values, or 0.0 for an empty ledger.",
       [],
       "assert average_amount([{'name':'a','amount':2},{'name':'b','amount':4}])==3\nassert average_amount([])==0.0",
       "return m.average_amount([{'name':'a','amount':2},{'name':'b','amount':4}])==3 and m.average_amount([])==0.0"),
    Op("names_list",
       "Write `names_list(ledger)` returning a list of the 'name' fields, in order, including duplicates.",
       [],
       "assert names_list([{'name':'a','amount':1},{'name':'a','amount':2}])==['a','a']",
       "return m.names_list([{'name':'a','amount':1},{'name':'a','amount':2}])==['a','a']"),
    Op("has_name",
       "Write `has_name(ledger, name)` returning True if any entry has that 'name', else False.",
       [],
       "assert has_name([{'name':'a','amount':1}],'a') is True\nassert has_name([],'a') is False",
       "return m.has_name([{'name':'a','amount':1}],'a') is True and m.has_name([],'a') is False"),
    Op("scale_amounts",
       "Write `scale_amounts(ledger, factor)` that multiplies every entry's 'amount' by factor in place and returns None.",
       [],
       "l=[{'name':'a','amount':2}]\nscale_amounts(l,3)\nassert l==[{'name':'a','amount':6}]",
       "l=[{'name':'a','amount':2}]; m.scale_amounts(l,3); return l==[{'name':'a','amount':6}]"),
    Op("positive_entries",
       "Write `positive_entries(ledger)` returning a list of entries whose 'amount' > 0, preserving order.",
       [],
       "assert positive_entries([{'name':'a','amount':-1},{'name':'b','amount':2}])==[{'name':'b','amount':2}]",
       "return m.positive_entries([{'name':'a','amount':-1},{'name':'b','amount':2}])==[{'name':'b','amount':2}]"),
    Op("negative_entries",
       "Write `negative_entries(ledger)` returning a list of entries whose 'amount' < 0, preserving order.",
       [],
       "assert negative_entries([{'name':'a','amount':-1},{'name':'b','amount':2}])==[{'name':'a','amount':-1}]",
       "return m.negative_entries([{'name':'a','amount':-1},{'name':'b','amount':2}])==[{'name':'a','amount':-1}]"),
    Op("rename_entry",
       "Write `rename_entry(ledger, old, new)` that sets 'name' to new for every entry whose 'name' equals old, in place. Returns None.",
       [],
       "l=[{'name':'a','amount':1}]\nrename_entry(l,'a','b')\nassert l==[{'name':'b','amount':1}]",
       "l=[{'name':'a','amount':1}]; m.rename_entry(l,'a','b'); return l==[{'name':'b','amount':1}]"),
    Op("min_entry",
       "Write `min_entry(ledger)` returning the entry with the smallest 'amount', or None if empty.",
       [],
       "assert min_entry([{'name':'a','amount':1},{'name':'b','amount':9}])=={'name':'a','amount':1}",
       "return m.min_entry([{'name':'a','amount':1},{'name':'b','amount':9}])=={'name':'a','amount':1} and m.min_entry([]) is None"),
    Op("clear_ledger",
       "Write `clear_ledger(ledger)` that removes all entries in place (leaving the same list object empty) and returns None.",
       [],
       "l=[{'name':'a','amount':1}]\nclear_ledger(l)\nassert l==[]",
       "l=[{'name':'a','amount':1}]; m.clear_ledger(l); return l==[]"),
]
OP_BY_NAME = {op.name: op for op in OP_BANK}

# Reference-correct implementations of ops that OTHER ops depend on. Used to SEED
# dependencies when an op is calibrated SOLO (Design §6): apply_fee calls add_entry and
# report calls total, so without their prerequisites present in the file, solo calibration
# fails artefactually (the bug that produced apply_fee p̂=0.000 / report p̂=0.800).
# Add a ref here for any op that appears in another op's `deps`.
REFS = {
    "add_entry": "def add_entry(ledger, name, amount):\n    ledger.append({'name': name, 'amount': amount})\n\n",
    "total": "def total(ledger):\n    return sum(e['amount'] for e in ledger)\n\n",
}

# ----------------------------------------------------------------------------- #
# Sandbox execution of the agent's module + a single step's held-out test.
# ----------------------------------------------------------------------------- #
GRADER_TEMPLATE = """\
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ledger", {modpath!r})
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except Exception as e:
    print("IMPORT_FAIL:" + repr(e)); sys.exit(0)
def check(m):
    {body}
try:
    ok = check(m)
    print("RESULT:" + ("PASS" if ok else "FAIL"))
except Exception as e:
    print("RESULT:FAIL:" + repr(e))
"""

def grade_step(modpath: str, op: Op, timeout=8) -> bool:
    body = "\n    ".join(op.held_out.strip().splitlines())
    script = GRADER_TEMPLATE.format(modpath=str(modpath), body=body)
    try:
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return "RESULT:PASS" in (out.stdout or "")

def run_visible(modpath: str, op: Op, timeout=8) -> str:
    """Run the agent-facing visible test; return a short pass/fail + any error."""
    check = textwrap.dedent(op.visible)
    script = (f"import importlib.util,sys\n"
              f"spec=importlib.util.spec_from_file_location('ledger',{str(modpath)!r})\n"
              f"m=importlib.util.module_from_spec(spec)\n"
              f"spec.loader.exec_module(m)\n"
              f"globals().update({{k:getattr(m,k) for k in dir(m) if not k.startswith('_')}})\n"
              f"{check}\nprint('VISIBLE_OK')\n")
    try:
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "tests timed out"
    if "VISIBLE_OK" in (out.stdout or ""):
        return "all visible tests passed"
    err = (out.stderr or out.stdout or "").strip().splitlines()
    return "FAILED: " + (err[-1] if err else "unknown error")

# ----------------------------------------------------------------------------- #
# Tool schemas (provider-neutral) + per-provider adapters.
# ----------------------------------------------------------------------------- #
TOOLS = [
    {"name": "append_function", "description": "Append a new Python function definition to ledger.py. Provide ONLY the new function's source code, nothing else.",
     "input_schema": {"type": "object",
                      "properties": {"code": {"type": "string"}},
                      "required": ["code"]}},
    {"name": "run_tests", "description": "Run the visible example test for the current task and return the result.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "done", "description": "Declare the current task finished.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "abstain", "description": "Declare you are unsure / likely to fail this task, and stop.",
     "input_schema": {"type": "object",
                      "properties": {"reason": {"type": "string"}},
                      "required": ["reason"]}},
]

def to_openai_tools(tools):
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}} for t in tools]


class LLM:
    """Thin adapter. Returns normalized: (text, [(tool_name, args, call_id)], usage_dict)."""
    def __init__(self, provider, model, temperature, abstain_enabled, cache=False):
        self.provider, self.model, self.temperature = provider, model, temperature
        self.cache = cache
        tools = TOOLS if abstain_enabled else [t for t in TOOLS if t["name"] != "abstain"]
        self.tools = tools
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI()
            self.otools = to_openai_tools(tools)
        else:
            raise ValueError("provider must be 'anthropic' or 'openai'")

    def step(self, history):
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model, max_tokens=1500, temperature=self.temperature,
                tools=self.tools, messages=history)
            text, calls = "", []
            for b in resp.content:
                if b.type == "text":
                    text += b.text
                elif b.type == "tool_use":
                    calls.append((b.name, b.input or {}, b.id))
            history.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
            u = resp.usage
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            # fold cached input into 'in' so tok/run still reflects total input billed
            usage = {"in": u.input_tokens + cr + cw, "out": u.output_tokens,
                     "cache_read": cr, "cache_write": cw}
            return text, calls, usage, resp.stop_reason
        else:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature,
                tools=self.otools, messages=history)
            msg = resp.choices[0].message
            calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    calls.append((tc.function.name, args, tc.id))
            history.append(msg.model_dump(exclude_none=True))
            u = resp.usage
            usage = {"in": u.prompt_tokens, "out": u.completion_tokens}
            return (msg.content or ""), calls, usage, resp.choices[0].finish_reason

    def tool_result_msg(self, call_id, content):
        if self.provider == "anthropic":
            return {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": content}]}
        return {"role": "tool", "tool_call_id": call_id, "content": content}


SYSTEM = ("You are a precise Python coding agent building a module `ledger.py` one function "
          "at a time. For each task, use append_function to add ONLY the new function "
          "(previously defined functions are already in the file — do not redefine them). "
          "You may call run_tests to check the visible example, then call done(). Call "
          "abstain(reason) only if you genuinely cannot complete the task correctly.")

# ----------------------------------------------------------------------------- #
# One step: agent works the current op until done/abstain/cap.
# ----------------------------------------------------------------------------- #
def sigs(modpath):
    """Signatures of functions already defined (compact context instead of full source)."""
    if not modpath.exists():
        return []
    return re.findall(r"^def (\w+\([^)]*\))", modpath.read_text(), re.M)

def run_step(llm, modpath: Path, op: Op, max_steps: int):
    if not modpath.exists():
        modpath.write_text("")
    available = ", ".join(sigs(modpath)) or "none yet"
    task = (f"TASK: {op.spec}\n\n"
            f"Functions already defined in ledger.py you may call: {available}.\n"
            f"Use append_function to add ONLY your new function (do not redefine existing ones). "
            f"Then call done().")
    history = ([{"role": "system", "content": SYSTEM}] if llm.provider == "openai" else [])
    if llm.provider == "anthropic":
        # System folded into first user msg; with --cache mark this prefix cacheable (Design §14).
        block = {"type": "text", "text": SYSTEM + "\n\n" + task}
        if getattr(llm, "cache", False):
            block["cache_control"] = {"type": "ephemeral"}
        history.append({"role": "user", "content": [block]})
    else:
        history.append({"role": "user", "content": task})

    tok_in = tok_out = 0
    for _ in range(max_steps):
        text, calls, usage, stop = llm.step(history)
        tok_in += usage["in"]; tok_out += usage["out"]
        if not calls:
            history.append({"role": "user", "content": "Use a tool, or call done() when finished."})
            continue
        terminal = None
        for (name, args, cid) in calls:
            if name == "append_function":
                code = args.get("code", "").strip()
                code = re.sub(r"^```(?:python)?\n?", "", code)  # strip stray markdown fences
                code = re.sub(r"\n?```$", "", code)
                with modpath.open("a") as fh:
                    fh.write("\n" + code.rstrip() + "\n")
                result = "appended"
            elif name == "run_tests":
                result = run_visible(modpath, op)
            elif name == "done":
                result = "ok"; terminal = "done"
            elif name == "abstain":
                result = "ok"; terminal = "abstain"
            else:
                result = "unknown tool"
            history.append(llm.tool_result_msg(cid, result))
        if terminal:
            return terminal, tok_in, tok_out
    return "maxsteps", tok_in, tok_out

# ----------------------------------------------------------------------------- #
# A coupled chain of length N: ops[0..N-1] in order, sharing the same module.
# Graded at the end with held-out tests. Records first-failing step + propagation.
# ----------------------------------------------------------------------------- #
@dataclass
class RunRecord:
    N: int
    outcome: str                 # success | wrong | abstained | maxsteps
    per_step_pass: list
    first_fail: int              # -1 if none
    propagated: bool
    abstained_at: int            # -1 if none
    tok_in: int
    tok_out: int

def run_chain(llm, ops, max_steps):
    with tempfile.TemporaryDirectory() as d:
        modpath = Path(d) / "ledger.py"
        terminals, tin, tout = [], 0, 0
        abstained_at = -1
        for i, op in enumerate(ops):
            term, ti, to = run_step(llm, modpath, op, max_steps)
            terminals.append(term); tin += ti; tout += to
            if term == "abstain":
                abstained_at = i; break
        # Grade every attempted step on the FINAL module state (coupling shows here).
        graded = ops if abstained_at < 0 else ops[:abstained_at]
        per_step = [grade_step(modpath, op) for op in graded]
        first_fail = next((i for i, ok in enumerate(per_step) if not ok), -1)
        # propagation: a failing step whose dependency also failed
        propagated = False
        name_to_idx = {op.name: i for i, op in enumerate(graded)}
        for i, op in enumerate(graded):
            if not per_step[i]:
                for dep in op.deps:
                    j = name_to_idx.get(dep)
                    if j is not None and not per_step[j]:
                        propagated = True
        if abstained_at >= 0:
            outcome = "abstained"
        elif all(per_step):
            outcome = "success"
        else:
            outcome = "wrong"
        return RunRecord(len(ops), outcome, per_step, first_fail, propagated,
                         abstained_at, tin, tout)

# ----------------------------------------------------------------------------- #
# Optional light calibration: each op SOLO (no chain), abstain disabled, to get p̂ᵢ.
# Parallelised across the n trials per op (independent runs).
# ----------------------------------------------------------------------------- #
def _calib_trial(llm, op, max_steps):
    with tempfile.TemporaryDirectory() as d:
        mp = Path(d) / "ledger.py"
        # §6: seed reference-correct dependencies so a SOLO op can call them.
        mp.write_text("".join(REFS.get(dep, "") for dep in op.deps))
        run_step(llm, mp, op, max_steps)
        # guard: if the agent overwrote the file and dropped a seeded dep, re-inject it
        content = mp.read_text()
        missing = "".join(REFS[dep] for dep in op.deps
                          if dep in REFS and f"def {dep}(" not in content)
        if missing:
            mp.write_text(missing + content)
        return grade_step(mp, op)

def calibrate(provider, model, temperature, n, max_steps, ops, cache=False, out=None, workers=5):
    llm = LLM(provider, model, temperature, abstain_enabled=False, cache=cache)
    phat = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for op in ops:
            futs = [ex.submit(_calib_trial, llm, op, max_steps) for _ in range(n)]
            passes = sum(1 for fu in futs if fu.result())
            phat[op.name] = passes / n
            print(f"  calib {op.name:16s} p̂={phat[op.name]:.3f}")
            if out is not None:
                out.write(json.dumps({"phase": "calib", "op": op.name,
                                      "phat": phat[op.name], "n": n}) + "\n"); out.flush()
    return phat

# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="cheap-but-capable model id (e.g. claude-sonnet-4-6, or an OpenAI mini)")
    ap.add_argument("--Ns", default="1,3,5,8,12,16,20")
    ap.add_argument("--runs", type=int, default=20, help="chain runs per N")
    ap.add_argument("--calib", type=int, default=20, help="solo runs per op for p̂ (0 = skip)")
    ap.add_argument("--max-steps", type=int, default=8, help="tool-loop cap per sub-task")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--calib-only", action="store_true", help="run calibration then stop")
    ap.add_argument("--cache", action="store_true", help="enable Anthropic prompt caching")
    ap.add_argument("--workers", type=int, default=5, help="concurrent API calls (lower if rate-limited)")
    args = ap.parse_args()

    Ns = [int(x) for x in args.Ns.split(",")]
    maxN = max(Ns)
    # Build the op sequence; cycle the bank with parametric repeats beyond its length.
    base = OP_BANK[:]
    ops_seq = [base[i % len(base)] for i in range(maxN)]
    if maxN > len(base):
        print(f"NOTE: N={maxN} exceeds distinct ops ({len(base)}); ops repeat beyond that.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"results_{stamp}.jsonl")

    phat = {}
    if args.calib > 0:
        print(f"== Calibration (solo, abstain off, deps seeded), n={args.calib} ==")
        with Path(f"calib_{stamp}.jsonl").open("w") as cf:
            phat = calibrate(args.provider, args.model, args.temperature,
                             args.calib, args.max_steps, base, cache=args.cache,
                             out=cf, workers=args.workers)
        print(f"Calibration records → calib_{stamp}.jsonl")
        if args.calib_only:
            print("\n--calib-only set; stopping after calibration.")
            return

    print(f"\n== Chains (abstain on), runs/N={args.runs} ==")
    llm = LLM(args.provider, args.model, args.temperature, abstain_enabled=True, cache=args.cache)
    summary = {}
    with out.open("w") as f:
        for N in Ns:
            ops = ops_seq[:N]
            recs = []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_chain, llm, ops, args.max_steps) for _ in range(args.runs)]
                for fut in as_completed(futs):
                    rec = fut.result()
                    recs.append(rec)
                    f.write(json.dumps(asdict(rec)) + "\n"); f.flush()
                    print(f"  N={N:2d} {len(recs):2d}/{args.runs} done ({rec.outcome})      ", end="\r")
            succ = sum(1 for x in recs if x.outcome == "success") / len(recs)
            abst = sum(1 for x in recs if x.outcome == "abstained") / len(recs)
            wrong = sum(1 for x in recs if x.outcome == "wrong") / len(recs)
            mx = sum(1 for x in recs if x.outcome == "maxsteps") / len(recs)
            toks = sum(x.tok_in + x.tok_out for x in recs) / len(recs)
            null = 1.0
            for op in ops:
                null *= phat.get(op.name, float("nan"))
            summary[N] = dict(success=succ, abstain=abst, wrong=wrong, maxsteps=mx,
                              mean_tokens=toks, null=null)
            print(" " * 40, end="\r")
            print(f"  N={N:2d}  success={succ:.2f}  Πp̂={null:.2f}  "
                  f"abstain={abst:.2f}  wrong={wrong:.2f}  maxsteps={mx:.2f}  ~{toks:.0f} tok/run")

    print(f"\nPer-run records → {out.resolve()}")
    print("\n== Summary (apply the §11 decision rules) ==")
    print(f"{'N':>3} {'success':>8} {'Πp̂(null)':>9} {'dev':>6} {'abstain':>8} {'tok/run':>8}")
    for N in Ns:
        s = summary[N]
        dev = (s["success"] - s["null"]) if s["null"] == s["null"] else float("nan")
        print(f"{N:>3} {s['success']:>8.2f} {s['null']:>9.2f} {dev:>+6.2f} "
              f"{s['abstain']:>8.2f} {s['mean_tokens']:>8.0f}")
    print(textwrap.dedent("""
        Reading it:
          • success ABOVE Πp̂  → error-recovery (agent absorbs slips)
          • success TRACKS Πp̂ → independent compounding (valid finding, proceed)
          • success BELOW Πp̂  → cascading/correlated failure (check first_fail + propagated)
          • FLAT (no decay by max N) → steps too easy / coupling too weak → revise before scaling
          • token/run growth across N → your main-study cost curve (≈quadratic in N expected)
    """))

if __name__ == "__main__":
    main()
