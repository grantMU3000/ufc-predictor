# Research — 2026-08-10: Data leakage taxonomy 

**Week:** 2 · **Curriculum area:** Data leakage taxonomy
**Time spent:** 60 min

## What I read/watched
- [Preventing training data leakage in AI systems](https://www.tonic.ai/blog/prevent-training-data-leakage-ai)

## Key ideas
- Training data/target leakage happens when information that wouldn't be available at the time of a prediction is used during model training.
- This leads to misleading performance metrics
- To prevent this, think about downstream implications while feature engineering
- Train-test contamination happens when data intended for evaluation influences model training
- Happens when a test set is included in preprocessing steps like normalization (Inflates performance metrics)
- Preprocessing leakage happens when global tranformation is applied to full datasets before it's split into train & test sets (Allows model to know about distribution of data that it shouldn't see during training)
- Commonly happens in early experimentation
- Improper data splitting (especially with time-series data) lets models train on signals that also appear in evaluation
- Some features may be available during inference, but they're highly correlated with a target & can lead to overfitting (Features should be flagged & reviewed)
- Use proper filteriing/time stamping to prevent overlap between training & target variables

- Keep logic, data sources, and labeling logic consistent throughout training
- If your model performs really well early on, even with minimal tuning, it could be an early sign of leakage
- Look for inadvertent interactions across train/test boundaries
- Run multiple tests removing high-risk features or re-splitting data under strict constraints & check performance
- Use automated tools that will audit your pipeline, monitor lineage, and test for reproducibility

## Questions / things I don't understand yet


## How this applies to my project
- Using temporal data, so use time-based data splits
- Ordering matters: Split data first, then fit scaler on training set, then apply fitted scaler to transform validation/test sets
- Split first, by date — train ≤2022, val 2023–24, test 2025+. This is a hard boundary, decided purely by fight date, nothing else.
-   Then, do row-level feature engineering (age, reach-to-height, etc.)
-   After that, do point-in-time career aggregates (Fit only on train)
