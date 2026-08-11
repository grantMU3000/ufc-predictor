# Research — 2026-08-11: Idempotent ETL patterns

**Week:** 1 · **Curriculum area:** idempotent ETL patterns

## What I read/watched
- [The Importance of Idempotency in Designing Data Pipelines](https://youtu.be/pKZ5n-y3ug4?si=OMCQNXy4RpNP57Fc)
- [Idempotent Pipelines: Build Once, Run Safely Forever](https://dev.to/alexmercedcoder/idempotent-pipelines-build-once-run-safely-forever-2o2o)
- [Dimensional data modeling and idempotent pipelines in 78 minutes with DataExpert.io](https://www.youtube.com/live/JeeqpK3o3LQ?si=-FCl8tsE0ufqM6wX)

## Key ideas
- You should never have any unexpected outputs
- When you reuse ingestion/migration it should always return the exact output you want
- Idempotency makes retries safe. Data pipelines can run the same job multiple times for a variety of reasons
- Partition overwrites deletes & recreate rows for a given segment. Best used for time-partitioned pipelines. For non-time-series data, this doesn't apply
- Upsert/MERGE pattern is used for Change data capture (CDC) workloads. Also good for slowly changing dimensions, and entity-centric data 
- Don't defer deduplication to a cleanup job because that makes the consumer have to deal with dirty data
- Run pipelines twice to ensure consistency
- Idempotent should be the same regardless of the day, hour, or # of times you run it
- Example: f(x) = 2x is alwayes the same output. Idempotent pipelines should perform like mathematical functions
- When using start_date, use an end_date to get a specific window of data. Efficiently keeps pipeline idempotent
- Think about idempotency from the beginning because it's hard & tedious to troubleshoot non-idempotent pipelines
- Slowly changing dimensions are a larger scale data problem (Not a problem with this project)
- Dimension change example: migrating from iPhone to Android

## Questions / things I don't understand yet


## How this applies to my project
- There should be no duplicate rows in my database
- I'll be running a pipeline that will handle a majority of rows that are duplicates, and none of these duplicate rows should be inserted
- Rows should be appropriately handle conflicts (Handle new bouts, change bouts from scheduled to completed, handle their results, etc.)
- We use upsert/merge to capture bout changes, and event changes. Ensures existing records update consistently
- Your pipeline & database should fail loudly that way you can catch errors