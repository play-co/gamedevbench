# GPT 6 Astra (High)

**229/333 tasks passed (68.77%; 68.8% rounded).** All 333 solver attempts completed; no tasks were skipped. The leaderboard uses the same normal-approximation 95% confidence interval as the existing entries: ±5.0 percentage points.

- Run date: September 7, 2026.
- Model: `gpt-6-astra`, reasoning effort `high`, Codex CLI 0.153.4.
- Godot 4.4.1; runtime-video enabled; screenshot MCP disabled.
- Strict confinement; 600-second solver timeout; 16 concurrent workers after the initial two-task check.
- Harness based on `983fc06`, with provider-stream timeout and Mesa rendering fixes in `2a15358`.

[final_results.json](final_results.json) contains the aggregate and per-task outcomes. Two initial attempts interrupted by a proxy timeout were repeated after the infrastructure fix. Task 0033 was retried after a provider capacity error; its completed retry failed validation. The other 332 completed results were preserved. Validation failures were not retried.

Ground-truth validation passed 331/333. Released tasks 0051 and 0052 have filename-case mismatches on Linux, also seen in the prior Sol High run. Task archives and validators were preserved for scoring; Astra passed task 0051 and failed task 0052. The latest harness adds strict confinement and prompt restrictions, so earlier leaderboard runs use different harness versions.

Cost fields use generic fallback pricing and are not verified Astra billing amounts.
