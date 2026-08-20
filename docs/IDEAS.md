- **Modern-only training window (~2010+)**: hypothesis that 1999-2010
  era fights (thin per-year volume, pre-professionalization per
  Saturday's KS-test era-drift findings) may be diluting rather than
  helping. Requires rerunning the full baseline suite (Elo/LR/LightGBM)
  on the restricted population for a fair before/after — not a
  same-day bolt-on. Good candidate for buffer week or a documented
  ablation in the model card.
  - **Increase the learning rate in LightGBM**: This comes at a risk of overfitting, but if we're training on a smaller dataset, it may be worth the risk in order to improve prediction accuracy. However, this could lead us to having to adjust ECE & log loss goals, which would make our model less professional.
  - **Have LightGBM do more trials/training sessions**: This may also lead to overfitting, but again, I think these models are worth at least testing to see if we can achieve a market-level model.
  - **Try XGBoost**: Less prone to overfitting. Uses more memory, but my dataset is relatively small