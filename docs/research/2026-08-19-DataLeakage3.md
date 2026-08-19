# Research — 2026-08-19: Data Leakage 3

**Week:** 2 · **Curriculum area:** Data Leakage Taxonomy 
**Time spent:** 30 min

## What I read/watched
- [A Solution to Leakage in Applied Machine Learning](https://builtin.com/articles/solution-leakage-applied-machine-learning)
- <source 2>

## Key ideas
- Even trained practitioners & experts can be subject to data leakage
- Most common leakage type is the training data interacting with the validation data
    - This is because the model will have access to evaluation data before it's actually evaluated
- Preprocessing should only be done on the training data set
    - If you're doing missing value imputation, compute sample means/medians only on the training set, and then use those to impute missing values in other data partitions (evaluation, etc.)
- Do feature selection/retrieval AFTER splitting train & evaluation sets
    - If you do this prior, you're leaking information about what works on evaluation data
- Sampling bias also causes leakage, so ensure you're picking a sample that's representative of the population you're predicting on
- Data leakage occurs when your model performs too optimistically in evaluations due to flaws in the machine learning pipeline

## Questions / things I don't understand yet


## How this applies to my project
I'll be auditing for leakage today, and the main worry is using future bouts/fighter outcomes to help with predictions/training. For a current bout, I won't have access to that information, so I shouldn't use it for training. 