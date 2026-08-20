Weekly database workflow:

1. Get stats from Greco & document date of updated CSV
2. Check/document row counts before ingesting
3. Run Greco ingest: uv run python -m data.ingestion.run_ingest
4. Run Wiki ingest: uv run python -m data.scraping.ingest_upcoming_events
5. Resolve fighter conflicts
6. Check row counts
7. Check bouts & bout stats
8. Run quality check
9. Document row counts again

Row Counts 08/17/26
(After Ingest)
bouts: 8687
bout stats: 41030
fighter aliases: 221
fighters: 4587
odds snapshots: 59972
predictions: 0
prediction results: 0

