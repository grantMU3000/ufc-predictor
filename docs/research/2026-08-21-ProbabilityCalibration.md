# Research — 2026-08-21: Probability Calibration

**Week:** 3 · **Curriculum area:** Probability calibration (Platt vs isotonic)
**Time spent:** 60 min

## What I read/watched
- [Probability Calibration in Machine Learning: Enhancing Model Usability](https://www.blog.trainindata.com/probability-calibration-in-machine-learning/)

## Key ideas
- Probability calibration helps a model's prediction accurately reflect the likelihood of an event occurring
- If a well-calibrated classification model predicts a 70% chance of an event occurring across multiple instances, we expect that event to occur 70% of the time
    - If a calibrated model assigns a 70% chance of rain for 100 different days, we expect it to rain on 70 days
- Logistic regression models typically output well-calibrated probabilities for classification tasks
- Random forests & SVMs require calibration, so you need to pick the appropriate model type
- Overly complex models tend to overfit, which leads to overconfidence and poorly calibrated predictions
- Imbalanced datasets can lead to biased probability estimates
- Knowing the confidence level of a prediction can inform critical decisions, so probability of a prediction can be just as important as the prediction itself
- Well-calibrated probabilities are essential for effective model ensembling
    - The reliability of each model's probability estimates affects the overall ensemble performance 
- Calibrated probabilities are more interpretable & trustworthy
- Probability calibration works by learning a mapping from the raw predictions to calibrated probabilities
    1. Obtain raw predictions
    2. Bining the predictions. Either dividing the prediction range into equal intervals, or groupint them based on equal-frequency. Ensures each bin contains an equal number of samples
    3. Calculate the true positive rate for each bin (Observed frequencies)
    4. Fitting a calibration model: Using either Platt Scaling or Isotonic Regression, A model is fit to map original predictions to observed frequencies
    5. Applying the calibration: The fitted calibration model transforms new predictions from the base prediction model
- Platt Scaling was developed for SVMs (Support Vector Machines), but is now used for various classifiers
    - It applies a logistic regression on the classifier's scores
    - Effective for neural networs & SVMs
    - Works well when the distribution in predicted probabilities is sigmoid-shaped
    - Better that isotonic if you have a small calibration dataset
- Isotonic Regression constrains predicted values to be monotonically increasing OR decreasing
    - It learns a piecewise constant function that maps the classifier's scores to calibrated probabilities
    - It's more flexible & can correct monotonic distortion
    - It may overfit on smaller datasets
    - Better for larger datasets (thousands of samples) and a complex distortion
- Brier score measures the mean squared difference between predicted probabilities and actual outcomes (Lower score implies better calibration)
- Log loss measures how well the predicted probabilities match true binary outcomes 
- Calibration curve plots the mean predicted probability against the true fraction of positive samples
    - A perfectly calibrated model would follow the diagonal line
    - x-axis represents the model's predicted probabilities, and y-axis shows actual proportion of positive samples for each bin
- Recommended approach is calibrating probabilities on a validation set (NOT training) with the fitted model, and evaluating it using the test set
- Cross-validation should always be used to prevent overfitting

## Questions / things I don't understand yet


## How this applies to my project
My model needs to be well calibrated in order to enable users to make accurate predictions. They can keep in mind what the model thinks in order to make pick selections.

Sounds like isotonic regression would be better