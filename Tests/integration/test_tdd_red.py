"""
apfel Integration Tests — TDD RED batch (branch tdd/red-tests-167-183)

DELIBERATELY FAILING tests, one per ticket that cannot be reached from the
pure-Swift unit target (see Package.swift: apfel-tests depends only on
ApfelCore + ApfelCLI, so executable-target bugs and server/CLI features are
red-tested here at the wire/CLI boundary).

These assert the CORRECT behaviour described in each GitHub issue. The fix that
makes them green is a SEPARATE follow-up task — do not implement here.

Covered (real assertions, red now):
  #167 json_schema, #169 prewarm (model-free), #170 apfel tag,
  #171 streamed structured output, #176 tool-def token undercount

Covered (Tier-3 loud placeholders — failure condition not externally
observable/triggerable, so a deterministic test needs a fix-phase testability
seam in the executable target; these pytest.fail rather than risk a false green):
  #168 top_p/greedy mapping, #175 summarize budget, #179 refusal accounting,
  #182 streaming-retry stdout

Model-dependent tests are FAILING, not skipped — consistent with the project's
"never skip" rule. They run under local `make test` where Apple Intelligence is
present. #169 is model-free and runs anywhere.

Run: python3 -m pytest Tests/integration/test_tdd_red.py -v
"""

import json
import pathlib
import subprocess

import httpx
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
BINARY = ROOT / ".build" / "release" / "apfel"
BASE = "http://localhost:11434"
BASE_URL = f"{BASE}/v1"
MODEL = "apple-foundationmodel"
TIMEOUT = 60


def _chat(payload):
    return httpx.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# #169 prewarm() — MODEL-FREE, runs everywhere (including GitHub CI)
# ---------------------------------------------------------------------------

def test_169_health_reports_prewarmed():
    """/health must expose whether the model was prewarmed at startup (#169).

    Contract chosen in the plan: GET /health returns a boolean "prewarmed"
    field. Today the field is absent -> RED.
    """
    data = httpx.get(f"{BASE}/health", timeout=10).json()
    assert "prewarmed" in data, "/health must include a 'prewarmed' field (#169)"
    assert isinstance(data["prewarmed"], bool)


# ---------------------------------------------------------------------------
# #167 response_format: json_schema — guaranteed structured outputs
# ---------------------------------------------------------------------------

def test_167_json_schema_conformance():
    """response_format json_schema must return content conforming to the schema (#167).

    Today only json_object is supported; json_schema is unsupported -> RED.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    resp = _chat({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Return a person named Alice aged 30."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Person", "schema": schema, "strict": True},
        },
    })
    assert resp.status_code == 200, f"json_schema request should succeed, got {resp.status_code}: {resp.text}"
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)  # must be valid JSON
    assert set(data.keys()) == {"name", "age"}, "output must conform exactly to the schema (no extra/missing keys)"
    assert isinstance(data["age"], int)


# ---------------------------------------------------------------------------
# #168 top_p honored + temperature:0 deterministic
# ---------------------------------------------------------------------------

def test_168_top_p_and_greedy_mapping():
    """#168: top_p must map to .random(probabilityThreshold:) and temperature:0
    to .greedy.

    Verified NOT externally observable: the on-device model is already
    empirically deterministic at temperature:0 for ordinary prompts (so an
    output-equality test is a false green), and top_p has no API-visible effect.
    The mapping lives in Session.makeGenerationOptions in the FoundationModels-
    coupled executable target, which the unit runner cannot import. A
    deterministic test needs the fix to expose makeGenerationOptions in a
    testable library. RED until then.
    """
    pytest.fail(
        "#168 RED placeholder: top_p/greedy mapping is not externally observable "
        "(temperature:0 is already deterministic here) and lives in the executable "
        "target; deterministic test needs the fix to expose makeGenerationOptions. "
        "Tracked on issue #168.")


# ---------------------------------------------------------------------------
# #170 apfel tag — contentTagging subcommand
# ---------------------------------------------------------------------------

def test_170_tag_subcommand_returns_tags_json():
    """`apfel tag -o json` must emit a JSON object with a 'tags' array (#170).

    Today there is no `tag` subcommand -> RED.
    """
    proc = subprocess.run(
        [str(BINARY), "tag", "-o", "json"],
        input="This local-first CLI tool is fast and private.",
        text=True, capture_output=True, timeout=TIMEOUT,
    )
    assert proc.returncode == 0, f"`apfel tag` should succeed, got rc={proc.returncode}, stderr={proc.stderr!r}"
    payload = json.loads(proc.stdout)
    assert "tags" in payload and isinstance(payload["tags"], list), "tag output must include a 'tags' array"


# ---------------------------------------------------------------------------
# #171 streamed structured output (PartiallyGenerated) — depends on #167
# ---------------------------------------------------------------------------

def test_171_streaming_json_schema_yields_valid_final_json():
    """Streaming with json_schema must produce a valid, conforming final JSON (#171)."""
    schema = {
        "type": "object",
        "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
        "required": ["steps"],
        "additionalProperties": False,
    }
    with httpx.stream(
        "POST", f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "List three steps to brew tea."}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "Steps", "schema": schema}},
            "stream": True,
        },
        timeout=TIMEOUT,
    ) as r:
        assert r.status_code == 200, f"streamed json_schema should succeed, got {r.status_code}"
        acc = ""
        for line in r.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunk = json.loads(line[len("data: "):])
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    acc += delta
    final = json.loads(acc)
    assert "steps" in final and isinstance(final["steps"], list)


# ---------------------------------------------------------------------------
# #176 fallback token counter ignores tool definitions (this machine = 26.3.1,
# so the buggy fallbackCount path is live)
# ---------------------------------------------------------------------------

def test_176_tool_definitions_count_toward_prompt_tokens():
    """A large tool definition must increase usage.prompt_tokens (#176).

    fallbackCount (macOS < 26.4) ignores Instructions.toolDefinitions entirely,
    so adding a huge tool schema barely changes prompt_tokens -> RED.
    """
    msg = [{"role": "user", "content": "Say hi."}]
    base = _chat({"model": MODEL, "messages": msg})
    assert base.status_code == 200, base.text
    base_pt = base.json()["usage"]["prompt_tokens"]

    big_desc = "This tool does an extremely elaborate calculation. " * 80  # ~4k chars
    withtool = _chat({
        "model": MODEL,
        "messages": msg,
        "tools": [{
            "type": "function",
            "function": {
                "name": "elaborate_calc",
                "description": big_desc,
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number", "description": big_desc}},
                    "required": ["x"],
                },
            },
        }],
    })
    assert withtool.status_code == 200, withtool.text
    tool_pt = withtool.json()["usage"]["prompt_tokens"]
    assert tool_pt > base_pt + 200, (
        f"large tool definition must add to prompt_tokens; base={base_pt}, with_tool={tool_pt} "
        "(fallbackCount ignores toolDefinitions)")


# ---------------------------------------------------------------------------
# #175 summarize strategy must not exceed the context budget
# ---------------------------------------------------------------------------

def test_175_summarize_keeps_prompt_within_budget():
    """#175: the summarize strategy must verify the final assembly fits the
    budget (unbounded summary + no final check can overflow).

    Verified NOT reliably triggerable at the wire boundary: whether the model
    emits a summary long enough to overflow the precise (budget - output
    reserve) window is non-deterministic, so a coarse prompt_tokens assertion is
    a false green. trimWithSummary lives in the executable target; a
    deterministic test needs the fix to make it injectable (stub summarizer ->
    assert the assembled entries fit budget). RED until then.
    """
    pytest.fail(
        "#175 RED placeholder: summarize budget overflow is non-deterministic at "
        "the wire boundary; deterministic test needs the fix to make trimWithSummary "
        "accept an injectable summarizer. Tracked on issue #175.")


# ---------------------------------------------------------------------------
# Tier 3 — failure condition is NOT externally triggerable. Explicit RED
# placeholders; a deterministic test needs the fix PR to add a testability seam
# (these bugs live in the FoundationModels-coupled executable target, which the
# unit runner cannot import, and the trigger — a mid-stream refusal / mid-stream
# retryable error — cannot be forced from a client).
# ---------------------------------------------------------------------------

def test_179_streaming_refusal_counts_pre_refusal_tokens():
    """#179: a refusal AFTER content has streamed must include the pre-refusal
    content in usage.completion_tokens.

    Not externally triggerable: a client cannot force the model to stream
    content and THEN refuse. The deterministic test requires extracting the
    refusal token-accounting into a pure helper (fix-phase seam). RED until then.
    """
    pytest.fail(
        "#179 RED placeholder: mid-stream refusal accounting is not externally "
        "triggerable; deterministic test needs the fix to extract a pure "
        "completion-token helper. Tracked on issue #179.")


def test_182_streaming_retry_prints_output_once():
    """#182: a retryable error mid-stream must not reprint already-streamed output.

    GREEN. The fix added the injectable seam this placeholder asked for:
    `StreamPrintSink` (Sources/Core/Chat/StreamRetryPolicy.swift) decouples the
    stdout side-effect from the retried streaming operation. One sink instance is
    shared across all `withRetry` attempts in `processPrompt`; it tracks a
    high-water mark of emitted characters and prints only the suffix beyond it,
    so a retry that re-streams the already-printed prefix emits nothing.

    The deterministic coverage lives in the pure-Swift unit suite, which can feed
    the sink a scripted failed-then-retried cumulative-snapshot sequence without
    the live model — see Tests/apfelTests/StreamPrintSinkTests.swift
    (runStreamPrintSinkTests), in particular
    "feed: a retry that re-streams the printed prefix does NOT reprint it (#182)".
    A mid-stream .rateLimited/.concurrentRequest error still cannot be forced from
    a client, so there is nothing left to assert at the wire boundary here.
    """
    pass
