# Decision Trees and Random Forests - AI & ML Internship Task 5

## Objective
Learn tree-based models for classification & regression.

## Tools Used
- Scikit-learn
- Graphviz (via scikit-learn's plot_tree)
- Matplotlib
- Pandas
- Numpy

## Dataset Used
Heart Disease Dataset from UCI Machine Learning Repository

## Files in this Repository
- `decision_tree_random_forest.py`: Main Python script implementing the task
- `decision_tree.png`: Visualization of the best decision tree (max_depth=2)
- `feature_importances.png`: Feature importances from Random Forest model
- `results.csv`: Summary of model accuracies
- `README.md`: This file

## Accomplishments
✅ Trained a Decision Tree Classifier and visualized the tree  
✅ Analyzed overfitting and controlled tree depth  
✅ Trained a Random Forest and compared accuracy  
✅ Interpreted feature importances  
✅ Evaluated using cross-validation  

## Results
- **Decision Tree Accuracy**: 0.7833
- **Best Decision Tree Accuracy (depth=2)**: 0.8667
- **Random Forest Accuracy**: 0.8833
- **Decision Tree CV Accuracy**: 0.7405 (+/- 0.0848)
- **Random Forest CV Accuracy**: 0.8078 (+/- 0.0752)

## Key Learnings
1. **Decision Trees**: Simple to understand and interpret, but prone to overfitting
2. **Overfitting Control**: Limiting tree depth helps prevent overfitting (optimal depth=2 for this dataset)
3. **Random Forests**: Ensemble method that reduces overfitting by averaging multiple decision trees
4. **Feature Importance**: Random Forest provides insights into which features are most predictive
5. **Cross-Validation**: Provides more robust estimate of model performance

## Interview Questions Preparation
1. **How does a decision tree work?**
   - A decision tree splits the data into subsets based on feature values, creating a tree-like model of decisions.
   - Each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label.

2. **What is entropy and information gain?**
   - Entropy measures the impurity or disorder in a set of examples.
   - Information gain measures the reduction in entropy achieved by splitting the data on a feature.
   - Decision trees use information gain to decide which feature to split on at each node.

3. **How is random forest better than a single tree?**
   - Random Forest reduces overfitting by creating multiple decision trees on different subsets of data and averaging their predictions.
   - It introduces randomness in feature selection, making trees less correlated and more robust.

4. **What is overfitting and how do you prevent it?**
   - Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on unseen data.
   - Prevention techniques: limiting tree depth, using ensemble methods (like Random Forest), cross-validation, and regularization.

5. **What is bagging?**
   - Bagging (Bootstrap Aggregating) is an ensemble technique where multiple models are trained on different subsets of the training data (created by sampling with replacement) and their predictions are averaged.
   - Random Forest is an extension of bagging that also uses random feature selection.

6. **How do you visualize a decision tree?**
   - Using libraries like scikit-learn's `plot_tree` or Graphviz to create a visual representation of the tree structure.
   - Each node shows the feature used for splitting, the threshold, and the samples/class distribution.

7. **How do you interpret feature importance?**
   - Feature importance indicates how much each feature contributes to the model's predictions.
   - In Random Forest, it's calculated as the average reduction in impurity (Gini importance) brought by each feature across all trees.

8. **What are the pros/cons of random forests?**
   - Pros: Reduces overfitting, handles high-dimensional data, provides feature importance, robust to outliers
   - Cons: Less interpretable than single trees, computationally intensive, can be slow to predict
