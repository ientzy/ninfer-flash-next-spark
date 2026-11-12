#!/usr/bin/env python3
"""Speed and correctness checks for a running ninfer server (standard library only).

    bench.py speed LABEL            five fixed prompts, greedy, one stream; saves runs/speed-LABEL.json
    bench.py compare A B            where two saved speed runs differ, and their tok/s
    bench.py copy                   a text repeated exactly (the best case for copy extension)
    bench.py streams N [N ...]      N greedy requests at once, total and per-stream tok/s
    bench.py edge LABEL             sampling with fixed seeds, penalties, JSON mode; saves runs/edge-LABEL.json
    bench.py compare-edge A B       two saved edge runs side by side
    bench.py mixed N                N requests at once, two of them JSON-constrained

Environment: BASE_URL (default http://127.0.0.1:8080), MODEL (default: the server's first model).
Speeds are the server's own decode rate (`timings.predicted_per_second`). One run is good to
about 5%; run the engine alone on the machine.
"""
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8080").rstrip("/")
RUNS = Path(__file__).resolve().parent / "runs"

FILE = "\n".join(
    f"def handler_{i}(request):\n    value = request.get('field_{i}', {i})\n    if value is None:\n"
    f"        raise ValueError('field_{i} is required')\n    return {{'status': 'ok', 'result': value * {i + 2}}}\n"
    for i in range(12))
PROMPTS = [
    ("new code", "Write a Python class for an LRU cache with get and put, with docstrings and a short usage "
                 "example."),
    ("explanation", "Explain in plain words how a hash table handles collisions, and when open addressing beats "
                    "chaining."),
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("file edit", "Rename the parameter `request` to `req` everywhere in this file and return the complete "
                  "updated file, nothing else.\n\n```python\n" + FILE + "```"),
    ("JSON", "Convert these notes to a JSON array of objects with fields name, qty, price: apples 4 at 1.20; "
             "pears 2 at 0.80; plums 12 at 0.25; figs 7 at 2.10; limes 9 at 0.40."),
]
JSON_PROMPT = ("Return a JSON object with keys city, country and three_facts (a list of three short strings) "
               "about Lisbon.")
COPY_TEXT = " ".join(f"Order {i}: {i * 37 % 1000} units of part {chr(65 + i % 26)}{i * 13 % 97} shipped to bay "
                     f"{i % 19} on day {i % 28 + 1}." for i in range(60))
EDGE = [
    ("sampled code", {"prompt": 0, "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 1234}),
    ("sampled explanation", {"prompt": 1, "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 77}),
    ("sampled story, temperature 1", {"prompt": 2, "temperature": 1.0, "top_p": 1.0, "top_k": 20, "seed": 5}),
    ("sampled story, min_p", {"prompt": 2, "temperature": 0.8, "min_p": 0.05, "top_k": 12, "seed": 9}),
    ("greedy, presence penalty", {"prompt": 2, "temperature": 0, "presence_penalty": 0.6}),
    ("sampled, frequency penalty", {"prompt": 1, "temperature": 0.7, "frequency_penalty": 0.3, "seed": 3}),
    ("JSON mode", {"content": JSON_PROMPT, "temperature": 0, "response_format": {"type": "json_object"}}),
]

_model = None


def model():
    global _model
    if _model is None:
        _model = os.environ.get("MODEL")
        if not _model:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
                _model = json.load(r)["data"][0]["id"]
    return _model


def chat(content, max_tokens=500, **extra):
    """One chat completion; returns text, tokens, tok/s and kept/drafted counts."""
    body = {"model": model(), "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False,
            "messages": [{"role": "user", "content": content}], **extra}
    request = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as r:
        reply = json.load(r)
    timings = reply.get("timings") or {}
    return {"text": reply["choices"][0]["message"]["content"] or "",
            "tokens": reply["usage"]["completion_tokens"],
            "tps": timings.get("predicted_per_second", 0.0),
            "kept": f"{timings.get('draft_n_accepted', 0)}/{timings.get('draft_n', 0)}",
            "finish": reply["choices"][0]["finish_reason"]}


def save(kind, label, data):
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{kind}-{label}.json").write_text(json.dumps(data, indent=1))


def load(kind, label):
    return json.loads((RUNS / f"{kind}-{label}.json").read_text())


def difference(a, b):
    if a == b:
        return "identical"
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"same for the first {n} of {len(a)}/{len(b)} characters"


def json_note(text):
    try:
        return "valid JSON: " + ", ".join(json.loads(text).keys())
    except (ValueError, AttributeError) as error:
        return f"INVALID JSON ({error})"


def at_once(jobs):
    """Run the callables together; returns their results in order and the wall time."""
    out = [None] * len(jobs)

    def run(i):
        out[i] = jobs[i]()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out, time.time() - start


def cmd_speed(label):
    rows = []
    for name, prompt in PROMPTS:
        r = chat(prompt)
        rows.append({"name": name, **r})
        print(f"{name:12} {r['tps']:6.1f} tok/s   {r['tokens']} tokens   drafts kept {r['kept']}")
    save("speed", label, rows)


def cmd_compare(kind, a, b):
    for x, y in zip(load(kind, a), load(kind, b)):
        print(f"{x['name']:30} {x['tps']:6.1f} -> {y['tps']:6.1f} tok/s   {difference(x['text'], y['text'])}")


def cmd_copy():
    r = chat("Repeat the following text exactly, with no changes and nothing added:\n\n" + COPY_TEXT,
             max_tokens=600)
    exact = COPY_TEXT.startswith(r["text"].strip())
    print(f"copy         {r['tps']:6.1f} tok/s   {r['tokens']} tokens   drafts kept {r['kept']}   "
          f"exact so far: {exact}")


def cmd_streams(counts):
    for n in counts:
        jobs = [lambda i=i: chat(PROMPTS[i % len(PROMPTS)][1]) for i in range(n)]
        out, wall = at_once(jobs)
        total = sum(o["tokens"] for o in out)
        print(f"{n} streams: {total / wall:6.1f} tok/s total ({total} tokens in {wall:.1f} s with prefill); "
              f"per stream {' '.join(f'{o['tps']:.0f}' for o in out)}")


def cmd_edge(label):
    rows = []
    for name, case in EDGE:
        case = dict(case)
        content = case.pop("content", None) or PROMPTS[case.pop("prompt")][1]
        r = chat(content, max_tokens=300, **case)
        note = "   " + json_note(r["text"]) if "response_format" in case else ""
        rows.append({"name": name, **r})
        print(f"{name:30} {r['tps']:6.1f} tok/s   {r['tokens']} tokens{note}")
    save("edge", label, rows)


def cmd_mixed(n):
    def job(i):
        if i in (1, 4):
            return chat(JSON_PROMPT, max_tokens=300, response_format={"type": "json_object"})
        return chat(PROMPTS[i % len(PROMPTS)][1], max_tokens=300)

    out, _ = at_once([lambda i=i: job(i) for i in range(n)])
    for i, r in enumerate(out):
        note = json_note(r["text"]) if i in (1, 4) else ""
        print(f"request {i}: {r['tps']:5.0f} tok/s   {r['finish']:6} {note}")


def main():
    args = sys.argv[1:]
    try:
        command = args[0]
        if command == "speed":
            cmd_speed(args[1])
        elif command == "compare":
            cmd_compare("speed", args[1], args[2])
        elif command == "compare-edge":
            cmd_compare("edge", args[1], args[2])
        elif command == "copy":
            cmd_copy()
        elif command == "streams":
            cmd_streams([int(a) for a in args[1:]] or [1, 2, 4, 8])
        elif command == "edge":
            cmd_edge(args[1])
        elif command == "mixed":
            cmd_mixed(int(args[1]) if len(args) > 1 else 6)
        else:
            raise IndexError
    except IndexError:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
