#!/usr/bin/env python3
"""
Extraction v2 Test Suite — runs all 10 synthetic scenarios against the extraction pipeline.

Usage:
    cd /data/projects/nanobot
    python experiments/extraction-v2/run_test_suite.py [--iteration N] [--scenario N] [--skip-dedup]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Add project root and this directory to path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

# Import core functions from the extraction script (same directory, hyphen in folder name)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("run_extraction", _HERE / "run_extraction.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

EXTRACTION_SYSTEM_PROMPT = _mod.EXTRACTION_SYSTEM_PROMPT
EXTRACTION_SYSTEM_PROMPT_V2 = _mod.EXTRACTION_SYSTEM_PROMPT_V2
EXTRACTION_SYSTEM_PROMPT_V3 = _mod.EXTRACTION_SYSTEM_PROMPT_V3
build_extraction_prompt = _mod.build_extraction_prompt
call_haiku = _mod.call_haiku
load_session_messages = _mod.load_session_messages
analyze_dedup = _mod.analyze_dedup

# Map iteration → system prompt to use
ITERATION_PROMPTS = {
    1: EXTRACTION_SYSTEM_PROMPT,
    2: EXTRACTION_SYSTEM_PROMPT_V2,
    3: EXTRACTION_SYSTEM_PROMPT_V3,
    4: EXTRACTION_SYSTEM_PROMPT_V3,  # V3 + taste signals
}

SCENARIOS_DIR = Path(__file__).parent / "test_scenarios"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Scenario loader
# ---------------------------------------------------------------------------

def load_scenario(n: int) -> dict[str, Any]:
    """Load scenario N: JSONL messages, pre-existing memories, and description."""
    jsonl_path = SCENARIOS_DIR / f"scenario_{n}.jsonl"
    memories_path = SCENARIOS_DIR / f"scenario_{n}_memories.json"
    desc_path = SCENARIOS_DIR / f"scenario_{n}_description.md"

    messages = load_session_messages(jsonl_path)
    memories = json.loads(memories_path.read_text(encoding="utf-8"))
    description = desc_path.read_text(encoding="utf-8") if desc_path.exists() else ""

    # Extract pass criteria from description
    pass_criteria = _parse_pass_criteria(description)
    expected_count_range = _parse_expected_count(description)

    return {
        "scenario_id": n,
        "messages": messages,
        "existing_memories": memories,
        "description": description,
        "pass_criteria": pass_criteria,
        "expected_count_range": expected_count_range,
    }


def _parse_pass_criteria(description: str) -> list[str]:
    """Extract pass criteria bullets from description markdown."""
    criteria = []
    in_pass_section = False
    for line in description.splitlines():
        if "## Pass Criteria" in line:
            in_pass_section = True
            continue
        if in_pass_section:
            if line.startswith("## "):
                break  # next section
            stripped = line.strip()
            if stripped.startswith("- "):
                criteria.append(stripped[2:])
    return criteria


def _parse_expected_count(description: str) -> tuple[int, int]:
    """Parse the expected extraction count range from description."""
    # Look for patterns like "(5-8 items)", "(2-3 items)", "(0-1 items)", "(10-12 items)"
    match = re.search(r"\((\d+)[-–](\d+) items", description)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Single count like "(exactly 1 item)"
    match = re.search(r"[Ee]xactly\s+(\d+) item", description)
    if match:
        n = int(match.group(1))
        return n, n
    # "Empty array" case
    if "Empty array" in description or "empty array" in description:
        return 0, 0
    # Default: 1-15
    return 1, 15


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

async def run_scenario(
    scenario: dict[str, Any],
    iteration: int,
    skip_dedup: bool = False,
) -> dict[str, Any]:
    """Run extraction on a single scenario and return results with quality scores."""
    n = scenario["scenario_id"]
    messages = scenario["messages"]
    existing_memories = scenario["existing_memories"]

    system_prompt = ITERATION_PROMPTS.get(iteration, EXTRACTION_SYSTEM_PROMPT_V2)

    print(f"\n{'='*60}")
    print(f"Scenario {n}: {_scenario_title(scenario['description'])}")
    print(f"  Messages: {len(messages)}, Existing memories: {len(existing_memories)}")

    # Use all messages as extraction window (no context split for test scenarios)
    context_msgs = []
    extraction_msgs = messages

    # Build prompt
    user_prompt = build_extraction_prompt(context_msgs, extraction_msgs, existing_memories)
    print(f"  Prompt size: {len(user_prompt)} chars (~{len(user_prompt)//4} tokens)")

    # Call Haiku
    t0 = time.monotonic()
    try:
        result = await call_haiku(system_prompt, user_prompt)
    except Exception as e:
        return {
            "scenario_id": n,
            "error": str(e),
            "passed": False,
            "items": [],
            "elapsed": round(time.monotonic() - t0, 2),
        }
    elapsed = time.monotonic() - t0

    print(f"  API elapsed: {result['elapsed_seconds']}s | Usage: {result['usage']}")

    # Parse JSON output
    raw = result["content"].strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    parse_error = None
    extracted_items = []
    try:
        extracted_items = json.loads(raw)
        if not isinstance(extracted_items, list):
            parse_error = f"Expected JSON array, got {type(extracted_items).__name__}"
            extracted_items = []
    except json.JSONDecodeError as e:
        parse_error = f"JSON parse error: {e}"
        # Attempt to extract partial array
        extracted_items = []

    print(f"  Extracted {len(extracted_items)} items" + (f" [PARSE ERROR: {parse_error}]" if parse_error else ""))

    # Dedup analysis
    deduped_items = extracted_items
    if not skip_dedup and existing_memories and extracted_items:
        try:
            print(f"  Running dedup analysis...")
            deduped_items = await analyze_dedup(extracted_items, existing_memories)
        except Exception as e:
            print(f"  Dedup failed: {e}")

    # Quality evaluation
    evaluation = evaluate_scenario(scenario, extracted_items, deduped_items, parse_error)

    print(f"  {'PASS' if evaluation['passed'] else 'FAIL'}: {evaluation['summary']}")
    for issue in evaluation.get("issues", []):
        print(f"    - {issue}")

    return {
        "scenario_id": n,
        "title": _scenario_title(scenario["description"]),
        "message_count": len(messages),
        "existing_memory_count": len(existing_memories),
        "extracted_count": len(extracted_items),
        "extracted_items": extracted_items,
        "deduped_items": deduped_items,
        "parse_error": parse_error,
        "api_elapsed": result["elapsed_seconds"],
        "api_usage": result["usage"],
        "evaluation": evaluation,
        "passed": evaluation["passed"],
        "prompt_size_chars": len(user_prompt),
    }


def _scenario_title(description: str) -> str:
    """Extract title from markdown description."""
    for line in description.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Unknown"


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

def evaluate_scenario(
    scenario: dict[str, Any],
    items: list[dict[str, Any]],
    deduped_items: list[dict[str, Any]],
    parse_error: str | None,
) -> dict[str, Any]:
    """Evaluate extraction quality against expected outcomes."""
    issues = []
    checks_passed = 0
    checks_total = 0

    def check(condition: bool, passed_msg: str, failed_msg: str) -> bool:
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            return True
        else:
            issues.append(failed_msg)
            return False

    # Parse error check
    check(parse_error is None, "Valid JSON output", f"JSON parse failed: {parse_error}")

    n = scenario["scenario_id"]
    count = len(items)
    lo, hi = scenario["expected_count_range"]
    desc = scenario["description"]

    # Count check
    check(
        lo <= count <= hi,
        f"Count {count} in expected range [{lo},{hi}]",
        f"Count {count} outside expected range [{lo},{hi}]",
    )

    # Scenario-specific checks
    if n == 1:
        # Clean extraction — check for key preferences
        contents = [i.get("content", "").lower() for i in items]
        full_text = " ".join(contents)
        check(any("fastapi" in c for c in contents), "FastAPI mentioned", "FastAPI decision not extracted")
        check(any("postgresql" in c or "postgres" in c for c in contents), "PostgreSQL mentioned", "PostgreSQL decision not extracted")
        check(any("uv" in c for c in contents), "uv preference captured", "uv package manager preference not extracted")
        check(any("fly.io" in c or "fly" in c for c in contents), "Fly.io deployment captured", "Fly.io deployment not extracted")
        check(any("pytest" in c for c in contents), "pytest captured", "pytest/testing preference not extracted")
        check(not any(i.get("action") not in ("add", "supersede", "reinforce") for i in items), "Valid actions", "Invalid action value found")

    elif n == 2:
        # Dedup — should supersede the model upgrade
        actions = [i.get("action") for i in items]
        contents = [i.get("content", "").lower() for i in items]
        full_text = " ".join(contents)
        check(
            any(a == "supersede" for a in actions) or any("haiku 4" in c or "haiku-4" in c or "4.5" in c for c in contents),
            "Model upgrade captured",
            "Model upgrade from 3.5→4.5 not captured"
        )
        check(
            count <= 3,
            f"Low count ({count}) avoids re-extracting existing memories",
            f"High count ({count}) suggests duplicating existing memories"
        )

    elif n == 3:
        # Supersede infrastructure migrations
        contents_lower = [i.get("content", "").lower() for i in items]
        full_text = " ".join(contents_lower)
        actions = [i.get("action") for i in items]
        supersede_count = sum(1 for a in actions if a == "supersede")
        check(supersede_count >= 3, f"At least 3 supersede actions (got {supersede_count})", f"Only {supersede_count} supersede actions, expected 4+")
        check("fly.io" in full_text or "fly" in full_text, "Fly.io migration captured", "Fly.io migration not captured")
        check("v3" in full_text or "version 3" in full_text, "API v3 migration captured", "API v3 migration not captured")
        check("neon" in full_text or "serverless" in full_text, "Neon DB migration captured", "Neon DB migration not captured")
        check("pgqueue" in full_text or "pg queue" in full_text or "redis" in full_text.replace("pgqueue", ""), "Queue migration captured", "Queue migration not captured")

    elif n == 4:
        # Noisy conversation — should be sparse
        contents = [i.get("content", "").lower() for i in items]
        full_text = " ".join(contents)
        check(count <= 4, f"Count {count} is sparse (noisy conversation)", f"Count {count} too high for noisy conversation")
        check("ios 16" in full_text or "ios16" in full_text or "ios" in full_text, "Platform minimums captured", "iOS 16 minimum not captured")
        check(
            not any("cursor" in c for c in contents),
            "Cursor tangent not extracted",
            "Off-topic Cursor discussion was extracted"
        )
        check(
            not any("cat" in c and "coffee" in c for c in contents),
            "Cat/coffee chat not extracted",
            "Casual cat/coffee chat was extracted"
        )

    elif n == 5:
        # Code-heavy — no code in output
        contents = [i.get("content", "") for i in items]
        full_text = " ".join(contents)
        has_code = any(
            snippet in full_text
            for snippet in ["def ", "import ", "return ", "class ", "ALGORITHM", "jwt.encode", "5 passed"]
        )
        check(not has_code, "No code content extracted", "Code content found in extracted items")
        contents_lower = [c.lower() for c in contents]
        check(any("rs256" in c or "asymmetric" in c for c in contents_lower), "RS256 decision captured", "RS256 algorithm decision not captured")
        check(count <= 4, f"Count {count} is focused", f"Count {count} too high for code-heavy session (over-extraction)")

    elif n == 6:
        # Contradictions — PostgreSQL and Kafka must NOT appear as final decisions
        contents_lower = [i.get("content", "").lower() for i in items]
        full_text = " ".join(contents_lower)
        check(
            "clickhouse" in full_text,
            "ClickHouse captured as final decision",
            "ClickHouse (final decision) not extracted"
        )
        check(
            "redpanda" in full_text,
            "Redpanda captured as final decision",
            "Redpanda (final decision) not extracted"
        )
        # Check PostgreSQL and Kafka aren't presented as chosen
        postgres_as_decision = any(
            ("postgresql" in c or "postgres" in c) and
            not ("not" in c or "ruled out" in c or "rejected" in c or "replaced" in c or "instead" in c)
            for c in contents_lower
        )
        kafka_as_decision = any(
            "kafka" in c and "redpanda" not in c and
            not ("not" in c or "ruled out" in c or "overkill" in c or "instead" in c)
            for c in contents_lower
        )
        check(not postgres_as_decision, "PostgreSQL NOT extracted as final decision", "PostgreSQL incorrectly extracted as final analytics DB choice")
        check(not kafka_as_decision, "Kafka NOT extracted as final decision", "Apache Kafka incorrectly extracted as final streaming choice")

    elif n == 7:
        # Multi-topic — all 4 topic areas should appear
        contents_lower = [i.get("content", "").lower() for i in items]
        full_text = " ".join(contents_lower)
        check("github actions" in full_text or "ghcr" in full_text or "fly.io" in full_text, "CI/CD topic covered", "CI/CD pipeline not extracted")
        check("next.js" in full_text or "nextjs" in full_text or "tailwind" in full_text or "shadcn" in full_text, "Frontend topic covered", "Frontend decisions not extracted")
        check("rate limit" in full_text or "100" in full_text or "1000" in full_text, "Rate limiting topic covered", "Rate limiting decisions not extracted")
        check("feature flag" in full_text or "launchdarkly" in full_text or "{team}" in full_text, "Feature flag topic covered", "Feature flag naming not extracted")

    elif n == 8:
        # Empty — must return []
        check(count == 0, "Empty array returned", f"Expected empty array, got {count} items")

    elif n == 9:
        # Dense decisions — high recall expected
        # Note: this scenario is a NEW platform, NOT the analytics stack from S6
        contents_lower = [i.get("content", "").lower() for i in items]
        full_text = " ".join(contents_lower)
        check("graphql" in full_text or "apollo" in full_text, "GraphQL captured", "GraphQL not captured")
        check("elasticsearch" in full_text, "Elasticsearch captured", "Elasticsearch not captured")
        check("vault" in full_text or "hashicorp" in full_text, "Vault secrets policy captured", "HashiCorp Vault not captured")
        check("terraform" in full_text, "Terraform captured", "Terraform not captured")
        check("sentry" in full_text, "Sentry captured", "Sentry not captured")
        check("cloudflare" in full_text or "r2" in full_text, "Cloudflare R2 captured", "Cloudflare R2 not captured")
        check("celery" in full_text, "Celery captured", "Celery not captured")

    elif n == 10:
        # Scope check — shared vs agent
        shared_items = [i for i in items if i.get("scope") == "shared"]
        agent_items = [i for i in items if i.get("scope") == "agent"]
        contents_lower = [i.get("content", "").lower() for i in items]
        full_text = " ".join(contents_lower)

        check(len(shared_items) >= 2, f"At least 2 shared-scope items (got {len(shared_items)})", f"Only {len(shared_items)} shared-scope items, expected ≥2")
        check(len(agent_items) >= 2, f"At least 2 agent-scope items (got {len(agent_items)})", f"Only {len(agent_items)} agent-scope items, expected ≥2")
        check("acme-corp" in full_text or "github.com" in full_text, "GitHub org captured", "GitHub org not captured")

        # Verify agent preferences aren't marked shared
        shared_contents_lower = [i.get("content", "").lower() for i in shared_items]
        style_in_shared = any(
            "short" in c or "direct" in c or "preamble" in c or "bullet" in c
            for c in shared_contents_lower
        )
        check(not style_in_shared, "Response style preference correctly scoped as agent (not shared)", "Response style preference incorrectly marked as shared scope")

        # Verify project facts aren't marked agent
        agent_contents_lower = [i.get("content", "").lower() for i in agent_items]
        project_fact_in_agent = any(
            "github.com/acme-corp" in c or "staging.acme-corp" in c or "on-call" in c or "2-week" in c
            for c in agent_contents_lower
        )
        check(not project_fact_in_agent, "Project facts correctly scoped as shared (not agent)", "Project facts incorrectly marked as agent scope")

    elif n == 11:
        # Taste extraction — UI corrections
        taste_items = [i for i in items if i.get("type") == "taste"]
        contents_lower = [i.get("content", "").lower() for i in items]
        taste_contents = [i.get("content", "").lower() for i in taste_items]
        full_text = " ".join(contents_lower)
        taste_text = " ".join(taste_contents)

        check(len(taste_items) >= 3, f"At least 3 taste items (got {len(taste_items)})", f"Only {len(taste_items)} taste items, expected ≥3")

        # Domain tags present
        has_domain_tags = any(
            c.startswith("[ui]") or c.startswith("[ux]") or "[ui]" in c or "[ux]" in c
            for c in taste_contents
        )
        check(has_domain_tags, "Domain tags present in taste items", "No domain tags ([ui], [ux]) found in taste items")

        # General "list = summary, modal = detail" pattern captured
        check(
            any("modal" in c and ("list" in c or "summary" in c or "detail" in c) for c in taste_contents),
            "List-vs-modal pattern captured as taste",
            "General list/modal pattern not captured as taste"
        )

        # Density preference (reduce spacing, keep fonts)
        check(
            any(("compact" in c or "dense" in c or "spacing" in c or "padding" in c or "whitespace" in c) for c in taste_contents),
            "Density/spacing preference captured as taste",
            "Density/spacing preference not captured"
        )

        # Rejected approach NOT extracted
        check(
            not any("12px" in c for c in contents_lower),
            "Rejected 12px font size not extracted",
            "Rejected 12px font size was extracted"
        )

        # Implementation states hidden from users
        check(
            any(("implementation" in c or "internal" in c or "draft" in c) and ("hide" in c or "not show" in c or "only show" in c or "shouldn" in c or "user" in c) for c in taste_contents)
            or any("deployed" in c and ("only" in c or "not" in c) for c in taste_contents),
            "Hide-implementation-states principle captured",
            "Hide-implementation-states principle not captured as taste"
        )

    elif n == 12:
        # Taste extraction — architecture & code review
        taste_items = [i for i in items if i.get("type") == "taste"]
        non_taste_items = [i for i in items if i.get("type") != "taste"]
        contents_lower = [i.get("content", "").lower() for i in items]
        taste_contents = [i.get("content", "").lower() for i in taste_items]
        non_taste_contents = [i.get("content", "").lower() for i in non_taste_items]
        full_text = " ".join(contents_lower)
        taste_text = " ".join(taste_contents)

        check(len(taste_items) >= 3, f"At least 3 taste items (got {len(taste_items)})", f"Only {len(taste_items)} taste items, expected ≥3")

        # Domain tags
        has_domain_tags = any(
            "[architecture]" in c or "[api]" in c or "[code-style]" in c or "[code_style]" in c
            for c in taste_contents
        )
        check(has_domain_tags, "Domain tags present in taste items", "No domain tags found in taste items")

        # Dependency principle is GENERALIZED (not just "don't use python-pushover")
        dep_principle = any(
            ("existing" in c or "check" in c or "before adding" in c or "already have" in c)
            and ("depend" in c or "package" in c or "util" in c or "library" in c)
            for c in taste_contents
        )
        check(dep_principle, "Dependency principle generalized as taste", "Dependency principle not generalized (or missing)")

        # API response structure as hard rule
        api_rule = any(
            ("json" in c or "status" in c) and ("response" in c or "api" in c or "endpoint" in c)
            for c in taste_contents
        )
        check(api_rule, "API response structure captured as taste", "API response structure rule not captured as taste")

        # Testing philosophy captured
        test_taste = any(
            ("test" in c and ("error" in c or "assert" in c or "response" in c or "body" in c or "structure" in c))
            for c in taste_contents
        )
        check(test_taste, "Testing philosophy captured as taste", "Testing philosophy not captured as taste")

        # Retry config should NOT be taste (it's a specific decision)
        retry_as_taste = any(
            "retry" in c and ("3" in c or "backoff" in c or "configurable" in c)
            for c in taste_contents
        )
        check(not retry_as_taste, "Retry config correctly NOT typed as taste", "Retry config incorrectly typed as taste (should be decision)")

        # Response helper location should NOT be taste
        helper_as_taste = any(
            "utils/responses" in c or "response helper" in c
            for c in taste_contents
        )
        check(not helper_as_taste, "Response helper location correctly NOT typed as taste", "Response helper location incorrectly typed as taste")

    score = checks_passed / checks_total if checks_total > 0 else 0.0
    passed = score >= 0.7 and parse_error is None  # require 70% of checks passing

    summary = f"{checks_passed}/{checks_total} checks passed (score={score:.0%})"
    return {
        "passed": passed,
        "score": score,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "summary": summary,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Extraction v2 test suite")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number for results directory")
    parser.add_argument("--scenario", type=int, default=None, help="Run only this scenario (1-10)")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip embedding-based dedup analysis")
    args = parser.parse_args()

    iteration_dir = RESULTS_DIR / f"iteration_{args.iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    scenarios_to_run = [args.scenario] if args.scenario else list(range(1, 13))

    print(f"\n{'='*60}")
    print(f"EXTRACTION v2 TEST SUITE — Iteration {args.iteration}")
    print(f"Scenarios: {scenarios_to_run}")
    print(f"Results: {iteration_dir}")
    print(f"{'='*60}")

    all_results = []
    total_tokens_in = 0
    total_tokens_out = 0
    suite_start = time.monotonic()

    for n in scenarios_to_run:
        scenario = load_scenario(n)
        result = await run_scenario(scenario, args.iteration, skip_dedup=args.skip_dedup)
        all_results.append(result)

        # Accumulate token usage
        usage = result.get("api_usage", {})
        total_tokens_in += usage.get("input_tokens", 0)
        total_tokens_out += usage.get("output_tokens", 0)

        # Save per-scenario result
        scenario_result_path = iteration_dir / f"scenario_{n}_result.json"
        scenario_result_path.write_text(
            json.dumps({**result, "extracted_items": result.get("extracted_items", [])}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    suite_elapsed = time.monotonic() - suite_start

    # Summary report
    passed = sum(1 for r in all_results if r.get("passed"))
    failed = len(all_results) - passed

    print(f"\n{'='*60}")
    print(f"TEST SUITE SUMMARY — Iteration {args.iteration}")
    print(f"{'='*60}")
    print(f"  Total scenarios: {len(all_results)}")
    print(f"  Passed: {passed} | Failed: {failed}")
    print(f"  Suite elapsed: {suite_elapsed:.1f}s")
    print(f"  Total tokens in: {total_tokens_in:,}")
    print(f"  Total tokens out: {total_tokens_out:,}")

    print(f"\n  Per-scenario results:")
    for r in all_results:
        status = "PASS" if r.get("passed") else "FAIL"
        score = r.get("evaluation", {}).get("score", 0)
        count = r.get("extracted_count", 0)
        title_short = r.get("title", f"Scenario {r['scenario_id']}")[:50]
        print(f"    [{status}] S{r['scenario_id']:02d}: {title_short}")
        print(f"           score={score:.0%}, extracted={count}, elapsed={r.get('api_elapsed','?')}s")
        for issue in r.get("evaluation", {}).get("issues", [])[:3]:
            print(f"           ⚠ {issue}")

    # Save full suite report
    report = {
        "iteration": args.iteration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total": len(all_results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(all_results) if all_results else 0,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "suite_elapsed_seconds": round(suite_elapsed, 1),
        },
        "scenarios": all_results,
    }

    report_path = iteration_dir / "suite_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Full report saved: {report_path}")

    # Print actionable failures for iteration
    failures = [r for r in all_results if not r.get("passed")]
    if failures:
        print(f"\n{'='*60}")
        print(f"FAILURES TO ADDRESS (Iteration {args.iteration})")
        print(f"{'='*60}")
        for r in failures:
            print(f"\n  Scenario {r['scenario_id']}: {r.get('title', '')}")
            for issue in r.get("evaluation", {}).get("issues", []):
                print(f"    - {issue}")

    return passed, failed, all_results


if __name__ == "__main__":
    asyncio.run(main())
