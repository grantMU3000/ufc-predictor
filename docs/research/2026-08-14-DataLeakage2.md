# Research — 2026-08-14: Data Leakage 2

**Week:** 2 · **Curriculum area:** Data Leakage Taxonomy 
**Time spent:** 60 min

## What I read/watched
- [Data Leakage Explained Visually](https://youtu.be/xzCXqYlV2HE?si=gL7gqmSM9qdOCwCx)
- [Data leakage in machine learning explained](https://www.educative.io/blog/what-is-data-leakage-in-machine-learning)

## Key ideas
Source 1
- Data leakage in loans can make a model that was intended to learn "which applicants are risky based on knowledge from today" use future information that won't be there in production for learning (Learning based on obvious signs instead of predicting in the present) 
-  The model is peeking at the future instead of predicting the future
- Target leakage is when you include features that heavily influence the outcome & isn't accessible in production (Learns to copy feature results)
- Don't normalize all data before splitting train/test data
    - This leads to data contamination since test data will be influencing the training data
    - Split first & preprocess training set only
    - Test set should be unseen until evaluation
- Treat data leakage like a pipeline moment. 
    - Define the prediction moment b/c it'll help with feature picking (Will X feature exist at prediction time?)
    - Watch for target proxies (If a feature is created as a result of the outcome, then it's probably a target)
    - Fit preprocessing on training data only. Then, transform both training & test data
    - Match your split to deployment
- Split should model the real world

Source 2
- Data leakage essentially allows the model to cheat during training
- Leakage is often hidden until the model is tested in production environments, so carefully designed training pipeline/validation strategies are important
- Preprocessing leakage can make the metrics become overly optimistic because the model is learning from testing data
- Maintain strict separation between training and evaluation data
- Feature selection should be performed after the train-test split
- Conduct feature reviews to ensure input variables don't contain information that reveals the target outcome
    - Feature importance analysis can reveal if a model relies heavily on variables that are suspiciously correlated with the target variable
    - Data validation pipelines ensures preprocessing steps occur in the correct order


## Questions / things I don't understand yet

## How this applies to my project
- Make sure you're only looking at UFC fighter information about a given fight from BEFORE said fight
- Ensure your training pipeline is valid (using as of date prevents most important leakage)
    - Have good validation strategies will prevent my model from making bad predictions on future events
