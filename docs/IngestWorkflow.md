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

Row Counts 08/23/26 (Post Greco Ingest)
bouts: 8687
bout_stats: 41082
events: 795
fighter aliases: 221
fighters: 4606
odds snapshots: 59972
predictions: 0
prediction results: 0

08/23/26 (Post Wiki Ingest)
bouts: 8707
bout_stats: 41082
events: 797 (Now 798 b/c I added a row for Fight Night on 10/31)
fighter aliases: 243
fighters: 4609
odds snapshots: 59972
predictions: 0
prediction results: 0

08/24/26 (Post Wiki Ingest fix)
bouts: 8707
bout_stats: 41082
events: 798
fighter aliases: 244 (Aline Pereira)
fighters: 4609
odds snapshots: 59972
predictions: 0
prediction results: 0

---

08/30/26 (Post-Greco Pull)
bouts: 8708
bout_stats: 41132
events: 798
fighter aliases: 244
fighters: 4616
odds snapshots: 59972
predictions: 0
prediction results: 0

08/30/26 (Post-Fighter Conflict fix)
bouts: 8708
bout_stats: 41132
events: 798
fighter aliases: 245 
fighters: 4613
odds snapshots: 59972
predictions: 0
prediction results: 0

08/30/26 (Wiki-API Pull & Resolution)
bouts: 8723
bout_stats: 41132
events: 798
fighter aliases: 260 
fighters: 4613
odds snapshots: 59972
predictions: 0
prediction results: 0