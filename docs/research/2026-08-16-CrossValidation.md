# Research — 2026-08-16: Time Series Cross-Validation

**Week:** 2 · **Curriculum area:** Temporal Cross Vailidation
**Time spent:** 30 min

## What I read/watched
- [How To Do Time Series Cross-Validation In Python](https://forecastegy.com/posts/time-series-cross-validation-python/)
- <source 2>

## Key ideas
- In a simple time split, you just pick a point in time, and everything that comes before it is used for training while everything else is validation
    - At least 50% of the data should be used for training
    - Do this before creating features or doing any preprocessing to ensure no future data is included in training
- Sliding window validation is when you take a point of time, train the data, and use the future time to validate. Then, this validated window is used for training, and the next window is used for validation
    - Example is training January, validating February, then training February & validating March
    - This is good for finding out how your model degrades with time
    - Can be used to train on the same window, and validating on multiple different validation sets
    - Closest to what's done in production
- Author starts out with simple time split, and then moves to the sliding window method afterward
- Expanding window validation slowly increases the size of the training window while keeping the validation set the same size
    - Good guardrail to prevent your model from looking too much at old patterns
    - This is used when you have a small amount of new data coming everytime
- Inserting a gap between the last training set & the first validation timestamp is useful when data is not available immediately
- Scikit-learn's TimeSeriesSplit function automates the expanding window method
    - Make sure your data is sorted before using the method
    -  The split method returns indices based on the amount of splits you do
    - Then, when you loop through the series, it's essentially the same as an expanding window

## Questions / things I don't understand yet


## How this applies to my project
- Simple time split sounds like the easiest method to use, but all methods seem useful