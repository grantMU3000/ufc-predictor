# Research — 2026-08-08: Cross-source Entity Resolution/Fuzzy Matching

**Week:** 1 · **Curriculum area:** Entity Resolution
**Time spent:** 55 min

## What I read/watched
- https://codecut.ai/rapidfuzz-rapid-string-matching-in-python/
- https://youtu.be/CbgO5KuCNic?si=iAccz_iTCFx37uRF

## Key ideas
- RapidFuzz is a fast string matching library that provides similarity metrics for fuzzy string matches. It's faster than FuzzyWuzzy
- It can be used for ratio comparison (Outputs a similarity score between strings)
- Process.extract will get the best matches to a string based on the metrics you desire (e.g. WRatio)
- RapidFuzz automatically handles case sensitivity and provides multiple matching algorithms
- Fuzzy matching handles typos & abbreviations 
- Embeddings can be used for additional context
- Data needs to be cleaned before it's outputted
- Fuzzy Matching is good for finding similar words while embeddings is best for assessing meaning with context
- Use Thresholds for baseline similarities for things to be considered matches

## Questions / things I don't understand yet


## How this applies to my project
- Will help with matching scraped event data to the fighters/events stored in my database
- I'll extend the fighter alias table with Wikipedia/ESPN source-key column