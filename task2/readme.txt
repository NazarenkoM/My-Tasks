1. Read the file.
2. Display descriptive statistical metrics. Since there's no understanding of data from zero or missing values, do nothing.
3. Display a correlation matrix for multicollinearity and scatter plot dependencies of features to the target. Conclude that feature 6 and 8 have a correlation of 0.94, and feature 6 correlates very well with the target.
4. Sort features by their information gain using the Random Forest model.
5. Feature 6 has the highest importance, while feature 8 is significantly less.
6. Build a Random Forest model for two datasets. The first uses all features except feature 8. The second uses only feature 6 and the target.
7. Compare cross-validation by RMSE and mean RMSE.
8. The model built on all parameters except feature 8 has the best performance:
   Cross-validated RMSE scores: [0.00383989 0.00378737 0.0037784  0.00377849 0.00379543]
   Mean RMSE: 0.003795917255571311
9. Compile the model in the `train.py` script.
10. Save the script for prediction as `predict.py`.
11. Save our prediction results.
