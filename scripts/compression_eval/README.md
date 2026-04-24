# compression_eval

Offline eval harness for `agent/context_compressor.py`. Runs a real
conversation transcript through the compressor, then probes the
compressed state with targeted questions graded on six dimensions.

**Status:** design only. See `DESIGN.md` for the full proposal and
open questions. `run_eval.py` is a placeholder.

## When to run

Before merging changes to:

- `agent/context_compressor.py`
- `agent/auxiliary_client.py` routing for compression tasks
- `agent/prompt_builder.py` compression-note phrasing

## Not for CI

This harness makes real model calls, costs ~$1 per run on a mainstream
model, takes minutes, and is LLM-graded (non-deterministic). It lives
in `scripts/` and is invoked by hand. `tests/` and
`scripts/run_tests.sh` do not touch it.

## Usage (once implemented)

```
python scripts/compression_eval/run_eval.py
python scripts/compression_eval/run_eval.py --fixtures=401-debug
python scripts/compression_eval/run_eval.py --runs=5 --label=my-prompt-v2
python scripts/compression_eval/run_eval.py --compare-to=results/2026-04-24_baseline
```

Results land in `results/<label>/report.md` and are intended to be
pasted verbatim into PR descriptions.

## Fixtures

Three scrubbed session snapshots live under `fixtures/`:

- `feature-impl-context-priority.json` — 75 msgs, investigate →
  patch → test → PR → merge
- `debug-session-feishu-id-model.json` — 59 msgs, PR triage +
  upstream docs + decision
- `config-build-competitive-scouts.json` — 61 msgs, iterative
  config accumulation (11 cron jobs)

Regenerate them from the maintainer's `~/.hermes/sessions/*.jsonl`
with `python3 scripts/compression_eval/scrub_fixtures.py`. The
scrubber pipeline and PII-audit checklist are documented in
`DESIGN.md` under **Scrubber pipeline**.

## Related

- `agent/context_compressor.py` — the thing under test
- `tests/agent/test_context_compressor.py` — structural unit tests
  that do run in CI
- `scripts/sample_and_compress.py` — the closest existing script in
  shape (offline, credential-requiring, not in CI)
