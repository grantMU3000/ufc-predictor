# Research — 2026-08-06: Idempotent ETL / upsert patterns in Postgres

**Week:** 1 · **Curriculum area:** idempotent ETL patterns
**Time spent:** 60 min

## What I read/watched
- https://queryplane.com/blog/postgres-upsert/

## Key ideas
- Postgres's UPSERT is handled by INSERT... ON CONFLICT (Used for "insert" or "update")
- DO UPDATE & DO NOTHING are the two variants put after the ON CONFLICT clause (UPDATE will insert if new, or replace the existing row if unique consstraint conflicts)
- DO NOTHING skips the insert silently if unique constraint conflicts
- ON CONFLICT (column) names a column that has a UNIQUE constraint or PRIMARY KEY
- DO UPDATE is convenient because it uses the EXCLUDED pseudotable (holds rows that couldn't be added) to update rows without having to manually set the row in a separate query
- If you want idempotent inserts, DO NOTHING is best because it's much faster than UPDATE, and it produces no error, row change, or side effect
- There are three different conflict targets to choose from, and picking the wrong one leads to UPSERT becoming a regular INSERT that fails, or succeeds and produces duplicate rows
- The targets are by column name; by multiple columns; by constraint name
- ON CONFLICT DO UPDATE needs a target. ON CONFLICT DO NOTHING is allowed without a target since it skips on any unique violation
- ON CONFLICT clause must match index definitions for updates
- A good pattern: Conditional updating. Only update when the new value is different/newer. Otherwise, you'd touch be touching the rows on every upsert
- Within these statements, ensure your keys are unique. Otherwise, this leads to errors (Deduplicate batch before sending it)
- unnest beats raw VALUES. It lets you pass column-typed arrays and avoid overhead
- RETURNING keyword is essentially a receipt of whatever Postgres actually did (Won't return anything if nothing happened during query)
- To make UPSERTS return on conflicts for DO NOTHINGs, you can either (a) use DO UPDATE and set the value to the pre-existing value (works, but Postgres touches the row when nothing changes, costing extra effort); (b) Essentially still DO NOTHING, but you then do a lookup to find that row and return that (Works & doesn't re-write, but looks up the row twice)
- Option 1 is better for high-write paths
- DO UPDATE on rows creates locks on a row while the update is happenning, so anyone else trying to update the row at the same time has to wait
- The fix is to batch the changes & do a big organized swap (I won't need this for now. Could become important if I add concurrent ingestion jobs, or high-frequency writes touching the same rows)

## Questions / things I don't understand yet
- For the 8 known name-collision fighters, does source_url fully protect me?


## How this applies to my project
- I'll have to update my database every week, and there'll be majority duplicate rows, so making sure the updating with the same data returns the same results is important
- RETURNING ids can help with future ingests where I use fighter IDs to update bout stats