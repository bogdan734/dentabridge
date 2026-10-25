# Clinical review of the demo urgency table

DentaBridge shows a "demo urgency hint" next to its image observations. The hint comes from a
small rule table, `packages/core/src/dentabridge_core/policy/urgency_v1.toml`, not from the model.
The table is **illustrative and not clinically reviewed** until a licensed dentist signs a review
record for the exact file. This page explains how a review is recorded.

## What a review covers

- Each enabled rule of the table: does the level (routine, soon, prompt) fit the finding it names?
- The patient wording in `packages/core/src/dentabridge_core/i18n/en.json` and `ru.json`.

A review is not clinical validation of the model and not a regulatory step. It only lets the
hint carry the tag "Reviewed by a dentist" instead of "Illustrative, not clinically reviewed".

## How to record one

1. Compute the fingerprint of the table file you reviewed:
   `shasum -a 256 packages/core/src/dentabridge_core/policy/urgency_v1.toml`
2. Save a JSON file in this folder, for example `surname-2026-11-20.json`:

```json
{
  "reviewer_name": "Dr Example",
  "qualification": "BDS",
  "registration": "000000",
  "reviewed_on": "2026-11-20",
  "table_sha256": "<the fingerprint from step 1>",
  "verdicts": {
    "R0": {"verdict": "agree", "comment": ""},
    "R2": {"verdict": "change", "comment": "Suggest prompt when two teeth are affected."},
    "R3": {"verdict": "agree", "comment": ""},
    "R9": {"verdict": "agree", "comment": ""}
  },
  "statement": "I reviewed this table and the patient wording for the demo."
}
```

3. A verdict is `agree`, `change` or `remove`, with a comment for the last two.

## When the tag changes

`review_status` returns "reviewed" only when a record names the sha256 of the table file in use,
gives a verdict for every enabled rule, and every verdict is `agree`. Any edit to the table, even
a comment, changes the fingerprint and returns the tag to illustrative until a new record is
signed. A malformed record is an error, never ignored.
