---
author: Robin
pubDatetime: 2026-04-06T08:05:00-03:00
title: "Building a Reusable SQL Adaptation Skill for Huawei Cloud GaussDB"
description: "A practical guide to turning SQL Server-to-GaussDB migration into a reusable engineering skill, with deterministic rewrite rules, search patterns, and operational guardrails."
tags:
  - gaussdb
  - huawei-cloud
  - sql-migration
  - sql-server
  - postgresql
  - database-modernization
  - engineering-productivity
  - migration-automation
featured: false
draft: false
---

# Building a Reusable SQL Adaptation Skill for Huawei Cloud GaussDB

## Why turn SQL migration into a skill?

Most SQL Server to GaussDB migrations do not fail because of one dramatic incompatibility. They fail because of repetition: hundreds of small T-SQL assumptions scattered across repositories, bulk loaders, queue consumers, and reporting queries. A team can fix those one by one, but that approach is slow, inconsistent, and hard to audit.

The better approach is to encode the migration workflow as a reusable skill:

- scan the codebase for known SQL Server patterns
- group findings by file and method
- prefer existing PostgreSQL or GaussDB implementations when they exist
- apply deterministic rewrite rules
- rescan only the touched files
- record the migration in a machine-readable report

That turns migration from an ad hoc effort into an operational capability.

## What the skill should contain

A practical GaussDB adaptation skill should include four parts:

1. A workflow document.
   This defines how to find migration scope, how to prioritize files, and how to validate changes.

2. A rulebook.
   This captures common SQL Server to GaussDB rewrite patterns with examples.

3. Search and reporting scripts.
   These accelerate discovery, especially when the migration source is an Excel remediation sheet or a large repository.

4. Guardrails.
   These tell engineers when not to mass-rewrite SQL and when to stop and do a manual conversion instead.

The key design principle is simple: use AI for search, grouping, and drafting, but keep the actual edits deterministic and reviewable.

## A practical migration workflow

The skill workflow can be summarized as:

1. Find the migration source of truth.
   This may be runtime errors, an `.xlsx` remediation sheet, or an existing PostgreSQL/GaussDB code path.

2. Build the file list first.
   Do not open random files. Extract file paths, line numbers, source SQL, and notes up front.

3. Search for the same pattern nearby.
   Excel sheets are often incomplete. A file with one `TOP` usually contains `NOLOCK`, `SCOPE_IDENTITY()`, or lock hints nearby.

4. Reuse existing GaussDB or PostgreSQL implementations first.
   If a sibling implementation already solved the same repository method, port that pattern instead of inventing a new one.

5. Rewrite by pattern group.
   Handle pagination, lock hints, identity retrieval, JSON expansion, bulk staging, and control-flow batches separately.

6. Rescan only the touched files.
   This keeps the feedback loop short and makes review cheaper.

7. Record every migration decision.
   A migration report is not optional. It becomes the playbook for the next repository.

## Core syntax conversion rules

The biggest value of the skill comes from standardizing the common rewrites.

### 1. `TOP` to `LIMIT`

SQL Server:

```sql
SELECT TOP (1) *
FROM pagamentos
WHERE status = 1;
```

GaussDB:

```sql
SELECT *
FROM pagamentos
WHERE status = 1
LIMIT 1;
```

Also:

- `TOP (@n)` becomes `LIMIT @n`
- queue selectors often need `ORDER BY ... LIMIT 1`, not just `LIMIT 1`

Why it matters:

- removes a parser incompatibility immediately
- makes pagination and row claiming portable

### 2. Remove `NOLOCK`

SQL Server:

```sql
SELECT *
FROM pagamentos WITH (NOLOCK);
```

GaussDB:

```sql
SELECT *
FROM pagamentos;
```

`NOLOCK` is usually not a performance optimization. It is often a consistency tradeoff disguised as one. In migration work, the safe default is to remove it.

Why it matters:

- avoids non-portable table hints
- prevents engineers from preserving dirty-read behavior without making that decision explicit

### 3. `OUTPUT INSERTED` and `SCOPE_IDENTITY()` to `RETURNING`

SQL Server:

```sql
INSERT INTO proc_carga (...)
VALUES (...);

SELECT CAST(SCOPE_IDENTITY() AS BIGINT);
```

GaussDB:

```sql
INSERT INTO proc_carga (...)
VALUES (...)
RETURNING cod_int_proc_carga;
```

And:

```sql
INSERT INTO notificacao (...)
OUTPUT INSERTED.cod_int_notificacao
VALUES (...);
```

becomes:

```sql
INSERT INTO notificacao (...)
VALUES (...)
RETURNING cod_int_notificacao;
```

Why it matters:

- reduces round trips
- avoids fragile sequence-name assumptions
- makes inserts safer under concurrency

### 4. SQL Server lock hints to `FOR UPDATE SKIP LOCKED`

A common SQL Server queue pattern is:

```sql
WITH cte AS (
    SELECT TOP (1) *
    FROM fila WITH (ROWLOCK, UPDLOCK, READPAST)
    WHERE status = 1
    ORDER BY id
)
UPDATE cte
SET status = 2;
```

The GaussDB pattern is usually:

```sql
UPDATE fila f
SET status = 2
FROM (
    SELECT id
    FROM fila
    WHERE status = 1
    ORDER BY id
    LIMIT 1
    FOR UPDATE SKIP LOCKED
) sel
WHERE f.id = sel.id
RETURNING f.*;
```

Why it matters:

- improves worker concurrency without hand-written lock hints
- avoids blocking cascades in multi-consumer queues
- gives a clear and portable “claim next item” pattern

### 5. `GETDATE()` and `SYSDATETIME()` to `CURRENT_TIMESTAMP`

SQL Server:

```sql
UPDATE pagamento
SET ctr_dth_atu = GETDATE();
```

GaussDB:

```sql
UPDATE pagamento
SET ctr_dth_atu = CURRENT_TIMESTAMP;
```

Why it matters:

- removes a trivial compatibility issue
- standardizes timestamp handling across repositories

### 6. `BIT` to boolean-compatible output

SQL Server often uses numeric boolean projections:

```sql
CAST(CASE WHEN d.nro_darc IS NULL THEN 0 ELSE 1 END AS BIT) AS existe
```

GaussDB:

```sql
CASE
    WHEN d.nro_darc IS NULL THEN FALSE
    ELSE TRUE
END AS existe
```

Why it matters:

- matches application models more naturally
- removes unnecessary casting ambiguity

### 7. `OPENJSON` to `json_to_recordset` or `json_array_elements_text`

SQL Server:

```sql
FROM OPENJSON(@Json)
WITH (
    NroDarc BIGINT '$.NroDarc',
    CodIntArquivoPagtoDet BIGINT '$.CodIntArquivoPagtoDet'
) j
```

GaussDB-style:

```sql
FROM json_to_recordset(@Json::json)
AS j(
    NroDarc BIGINT,
    CodIntArquivoPagtoDet BIGINT
)
```

Why it matters:

- removes reliance on SQL Server JSON syntax
- often produces simpler, more explicit set-based SQL

### 8. `@@ROWCOUNT` control flow to `UPDATE ... RETURNING` plus CTEs

SQL Server:

```sql
UPDATE t
SET status = 2
WHERE id = @Id;

IF @@ROWCOUNT <= 0
BEGIN
    SELECT -1;
END
ELSE
BEGIN
    INSERT INTO log (...) VALUES (...);
END
```

GaussDB:

```sql
WITH upd AS (
    UPDATE t
    SET status = 2
    WHERE id = @Id
    RETURNING id
),
ins AS (
    INSERT INTO log (...)
    SELECT ...
    FROM upd
    RETURNING id
)
SELECT COALESCE((SELECT id FROM ins), -1);
```

Why it matters:

- keeps logic set-based
- avoids procedural T-SQL fragments inside repository code
- improves readability under review

## Bulk loading and temp table adaptation

In practice, bulk pipelines are where the biggest migration pain shows up.

Common SQL Server staging patterns include:

- `#temp` tables
- `SqlBulkCopy`
- `WITH (TABLOCK)`
- SQL Server-specific staging column types like `DATETIME2`, `DATETIMEOFFSET`, `UNIQUEIDENTIFIER`, and `TINYINT`

Typical conversions:

- `#stg_table` -> `CREATE TEMP TABLE stg_table`
- `UNIQUEIDENTIFIER` -> `UUID`
- `DATETIMEOFFSET(3)` -> `TIMESTAMPTZ(3)`
- `DATETIME2(3)` -> `TIMESTAMP(3)`
- `TINYINT` -> `SMALLINT`
- remove `WITH (TABLOCK)`

This is more than syntax cleanup. It changes how bulk pipelines behave under real workload.

Why it matters:

- temp tables become provider-compatible
- bulk writers become reusable across repositories
- schema cleanup reduces coupling to SQL Server assumptions

## Where the performance value comes from

Migration is usually framed as compatibility work, but the high-value rewrites also improve runtime behavior.

### Fewer round trips

`RETURNING` removes the extra “insert then fetch identity” query pattern. This matters most in hot insert paths.

### Better concurrent consumers

Replacing `ROWLOCK/UPDLOCK/READPAST` work-queue logic with `FOR UPDATE SKIP LOCKED` makes contention behavior clearer and usually better under load. It reduces accidental blocking between workers and makes horizontal scaling more predictable.

### Less procedural SQL in the application layer

Moving away from `@@ROWCOUNT`, `TRY/CATCH`, and large T-SQL batches toward CTE-based SQL or database functions reduces repository complexity. That lowers maintenance cost and review time.

### Cleaner bulk ingestion paths

A dedicated GaussDB bulk dialect gives the team one place to optimize:

- batch size
- temp table naming
- sequence allocation
- parameterized multi-row insert strategy

That is much easier to tune than dozens of repository-specific ad hoc implementations.

### Better operational safety

Removing SQL Server-specific hints often eliminates “it compiles but behaves differently” failures. That does not show up as raw benchmark gain, but it prevents many of the worst migration regressions.

## What not to automate blindly

A good skill should be opinionated about risk.

Do not mass-rewrite these with regex alone:

- `MERGE`
- dynamic table names
- complex T-SQL `BEGIN TRY / BEGIN CATCH`
- table variables
- mixed insert-update procedural batches
- sequence allocation code with correctness requirements

These patterns need either:

- a CTE rewrite
- a database-side function or procedure
- or a manual redesign of the repository method

## How AI helps without becoming dangerous

AI is most valuable in three places:

1. Discovery.
   It can group SQL findings by file and identify related patterns nearby.

2. Drafting.
   It can propose the first-pass rewrite from a known rule set.

3. Reporting.
   It can generate migration worklists and explain why each rewrite happened.

AI is least trustworthy when asked to blindly rewrite an entire codebase with no local context. The right model is assisted migration, not unsupervised migration.

## Recommended search patterns

These searches find most SQL Server-specific fragments quickly:

```bash
rg -n 'TOP \(|NOLOCK|OUTPUT INSERTED|SCOPE_IDENTITY|OPENJSON|@@ROWCOUNT|ROWLOCK|UPDLOCK|READPAST|GETDATE\(|SYSDATETIME\(' <target-dir>
rg -n 'FOR UPDATE SKIP LOCKED|RETURNING|json_to_recordset|json_array_elements_text' <target-dir>
rg -n 'MERGE\s|OPTION \(MAXDOP|DATETIMEOFFSET|DATETIME2|UNIQUEIDENTIFIER|TINYINT|#stg_' <target-dir>
```

These are simple, but they are effective because most migration work is repetitive.

## A minimal implementation blueprint

If you want to add this skill to your engineering toolkit, the smallest useful version should include:

- `SKILL.md`
  workflow, search patterns, validation steps

- `gaussdb_rules.md`
  rewrite rules and examples

- `extract_xlsx_sql_mappings.py`
  turns remediation spreadsheets into actionable file lists

- `build_migration_worklist.py`
  scans a repository and produces a deterministic worklist

- `render_migration_report.py`
  converts the worklist into a reviewable Markdown report

That is enough to make migration repeatable across multiple repositories.

## Final takeaway

The highest-value part of a GaussDB migration is not the individual SQL fix. It is the reusable adaptation skill behind it.

Once the rules are explicit, the team stops rediscovering the same incompatibilities:

- `TOP` becomes `LIMIT`
- `NOLOCK` disappears
- `OUTPUT INSERTED` becomes `RETURNING`
- queue locking becomes `FOR UPDATE SKIP LOCKED`
- `OPENJSON` becomes `json_to_recordset`
- procedural row-count logic becomes set-based SQL

That is where the real payoff is:

- faster migrations
- fewer production surprises
- cleaner review cycles
- better concurrency behavior
- and a much lower cost for the next database transition
