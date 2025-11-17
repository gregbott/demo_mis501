import marimo

__generated_with = "0.16.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # Random Forests in Machine Learning

    This notebook explores Random Forests, a powerful ensemble learning algorithm that combines multiple decision trees to create more robust and accurate predictions.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import make_moons, fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report
    return (
        RandomForestClassifier,
        RandomForestRegressor,
        accuracy_score,
        fetch_california_housing,
        make_moons,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## What is a Random Forest?

    A **Random Forest** is an ensemble learning algorithm that builds multiple decision trees and combines their predictions.

    **Key Characteristics:**

    1. **Ensemble Method** - Instead of relying on a single decision tree (which can overfit), a random forest trains many trees and aggregates their predictions

    2. **Bootstrap Aggregating (Bagging)** - Each tree is trained on a random subset of the training data, sampled with replacement. This creates diversity among trees

    3. **Random Feature Selection** - At each split in each tree, only a random subset of features is considered. This further increases diversity

    4. **Voting/Averaging**:
       - **Classification**: Each tree votes for a class, and the majority vote wins (mode)
       - **Regression**: The predictions from all trees are averaged

    5. **Reduces Overfitting** - By combining multiple trees trained on different data samples, random forests reduce variance and overfitting

    **Advantages:**
    - High accuracy and robustness
    - Handles both classification and regression
    - Works well with large datasets
    - Can handle non-linear relationships
    - Provides feature importance rankings
    - Less prone to overfitting than single decision trees

    **Disadvantages:**
    - More computationally expensive than single trees
    - Less interpretable than a single decision tree
    - Can be slow to predict on large datasets
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Example 1: Classification with make_moons Dataset

    The `make_moons()` dataset creates two interleaving half circles - a non-linearly separable dataset. This is an excellent example to show how random forests handle complex decision boundaries.
    """
    )
    return


@app.cell
def _(make_moons, plt, train_test_split):
    # Generate the make_moons dataset
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
        X_moons, y_moons, test_size=0.2, random_state=42
    )

    # Visualize the dataset
    moons_fig, moons_ax = plt.subplots(figsize=(10, 6))
    moons_ax.scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], c='blue', label='Class 0', alpha=0.6)
    moons_ax.scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], c='red', label='Class 1', alpha=0.6)
    moons_ax.set_xlabel("Feature 1")
    moons_ax.set_ylabel("Feature 2")
    moons_ax.set_title("Make Moons Dataset - Non-linear Classification")
    moons_ax.legend()
    moons_ax.grid(True, alpha=0.3)
    return (
        X_moons,
        X_test_moons,
        X_train_moons,
        moons_fig,
        y_moons,
        y_test_moons,
        y_train_moons,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Dataset Visualization""")
    return


@app.cell
def _(moons_fig):
    moons_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Train Random Forest Classifier""")
    return


@app.cell
def _(RandomForestClassifier, X_train_moons, y_train_moons):
    # Train a random forest classifier
    rf_moons = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_moons.fit(X_train_moons, y_train_moons)
    return (rf_moons,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Make Predictions and Evaluate""")
    return


@app.cell
def _(X_test_moons, accuracy_score, rf_moons, y_test_moons):
    # Make predictions
    y_pred_moons = rf_moons.predict(X_test_moons)

    # Calculate accuracy
    moons_accuracy = accuracy_score(y_test_moons, y_pred_moons)
    return (moons_accuracy,)


@app.cell(hide_code=True)
def _(mo, moons_accuracy):
    mo.md(
        f"""
    **Accuracy on Test Set: {moons_accuracy:.4f}** ({moons_accuracy * 100:.2f}%)

    The random forest successfully learned the non-linear decision boundary between the two moons.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Visualize Decision Boundary""")
    return


@app.cell
def _(X_moons, np, plt, rf_moons, y_moons):
    # Create a mesh to plot the decision boundary
    h = 0.02
    x_min, x_max = X_moons[:, 0].min() - 0.1, X_moons[:, 0].max() + 0.1
    y_min, y_max = X_moons[:, 1].min() - 0.1, X_moons[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = rf_moons.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    boundary_fig, boundary_ax = plt.subplots(figsize=(10, 6))
    boundary_ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r')
    boundary_ax.scatter(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], c='blue', label='Class 0', alpha=0.6, edgecolors='k', linewidth=0.5)
    boundary_ax.scatter(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], c='red', label='Class 1', alpha=0.6, edgecolors='k', linewidth=0.5)
    boundary_ax.set_xlabel("Feature 1")
    boundary_ax.set_ylabel("Feature 2")
    boundary_ax.set_title("Random Forest Decision Boundary on Make Moons")
    boundary_ax.legend()
    boundary_ax.grid(True, alpha=0.3)
    return (boundary_fig,)


@app.cell
def _(boundary_fig):
    boundary_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Feature Importance in Make Moons

    Looking at the code, Feature 1 and Feature 2 are synthetic numerical 
      features generated by the make_moons() function - they're not named
      features with real-world meaning.

      Specifically:
      - Feature 1 is the x-coordinate of each point (ranging roughly from -0.1
      to 1.1)
      - Feature 2 is the y-coordinate of each point (ranging roughly from -0.5
      to 1.0)

      The make_moons() function creates a synthetic dataset where:
      X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

      This generates 300 random points in 2D space arranged in two interleaving
      half-moon shapes. The function returns:
      - X_moons: An array of shape (300, 2) where each row is a point with two
      coordinates
      - y_moons: Class labels (0 or 1) indicating which moon each point belongs
      to

      So when you see the scatter plot:
      - The x-axis is "Feature 1" (the first coordinate)
      - The y-axis is "Feature 2" (the second coordinate)
      - The red and blue dots show the two classes

      These are purely artificial features used for demonstration - the point is
       to show how Random Forests can learn a non-linear decision boundary that
      separates these two interleaving moon shapes, even though the features
      themselves don't have any special meaning beyond being x and y
      coordinates.
    """
    )
    return


@app.cell
def _(plt, rf_moons):
    importance_moons_fig, importance_moons_ax = plt.subplots(figsize=(8, 5))
    moons_importances = rf_moons.feature_importances_
    importance_moons_ax.barh(["Feature 1", "Feature 2"], moons_importances)
    importance_moons_ax.set_xlabel("Importance")
    importance_moons_ax.set_title("Feature Importance in Random Forest (Make Moons)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Example 2: Regression with California Housing Dataset

    The California Housing dataset contains real estate prices and related features. We'll use a Random Forest Regressor to predict house prices, demonstrating how random forests handle continuous regression problems.
    """
    )
    return


@app.cell
def _(fetch_california_housing, pd, train_test_split):
    # Fetch California housing dataset
    california = fetch_california_housing()
    X_housing = california.data
    y_housing = california.target
    feature_names_housing = california.feature_names

    # Create a DataFrame for better visibility
    housing_df = pd.DataFrame(X_housing, columns=feature_names_housing)
    housing_df['Price'] = y_housing

    # Split the data
    X_train_housing, X_test_housing, y_train_housing, y_test_housing = train_test_split(
        X_housing, y_housing, test_size=0.2, random_state=42
    )
    return (
        X_test_housing,
        X_train_housing,
        feature_names_housing,
        housing_df,
        y_test_housing,
        y_train_housing,
    )


@app.cell(hide_code=True)
def _(housing_df, mo):
    mo.md(
        f"""
    ### Dataset Overview

    **Shape**: {housing_df.shape}
    **Features**: {', '.join(housing_df.columns[:-1])}

    The dataset contains information about California housing blocks with median prices.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Feature Descriptions

    **MedInc** - Median income in block group (in tens of thousands of dollars)
    - Example: 8.3 means median income is $83,000

    **HouseAge** - Median house age in the block group (in years)
    - How old the typical house is in that area

    **AveRooms** - Average number of rooms per household
    - Total rooms divided by total households

    **AveBedrms** - Average number of bedrooms per household
    - Total bedrooms divided by total households

    **Population** - Total population in the block group
    - Number of residents in that census block group

    **AveOccup** - Average occupancy rate (people per household)
    - Average household size

    **Latitude** - Latitude coordinate of the block group center
    - North-South position (higher = further north)

    **Longitude** - Longitude coordinate of the block group center
    - East-West position (higher = further east)

    **Price** - Median house value in the block group (in hundreds of thousands of dollars)
    - **Target variable** we're trying to predict
    - Example: 4.526 means median price is $452,600
    """
    )
    return


@app.cell
def _(housing_df):
    housing_df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Train Random Forest Regressor""")
    return


@app.cell
def _(RandomForestRegressor, X_train_housing, y_train_housing):
    # Train a random forest regressor
    rf_housing = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20)
    rf_housing.fit(X_train_housing, y_train_housing)
    return (rf_housing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Make Predictions and Evaluate""")
    return


@app.cell
def _(
    X_test_housing,
    mean_squared_error,
    np,
    r2_score,
    rf_housing,
    y_test_housing,
):
    # Make predictions
    y_pred_housing = rf_housing.predict(X_test_housing)

    # Calculate metrics
    mse_housing = mean_squared_error(y_test_housing, y_pred_housing)
    rmse_housing = np.sqrt(mse_housing)
    r2_housing = r2_score(y_test_housing, y_pred_housing)
    return r2_housing, rmse_housing, y_pred_housing


@app.cell(hide_code=True)
def _(mo, r2_housing, rmse_housing):
    mo.md(
        f"""
    **Model Performance:**
    - R² Score: {r2_housing:.4f} (explains {r2_housing * 100:.2f}% of variance)
    - RMSE: ${rmse_housing:.4f}M (Root Mean Squared Error)

    The random forest regressor explains most of the variance in housing prices, showing its effectiveness for this regression task.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Actual vs Predicted Prices""")
    return


@app.cell
def _(plt, y_pred_housing, y_test_housing):
    pred_fig, pred_ax = plt.subplots(figsize=(10, 6))
    pred_ax.scatter(y_test_housing, y_pred_housing, alpha=0.5)
    pred_ax.plot([y_test_housing.min(), y_test_housing.max()], [y_test_housing.min(), y_test_housing.max()], 'r--', lw=2)
    pred_ax.set_xlabel("Actual Price ($M)")
    pred_ax.set_ylabel("Predicted Price ($M)")
    pred_ax.set_title("Random Forest Predictions vs Actual Housing Prices")
    pred_ax.grid(True, alpha=0.3)
    return (pred_fig,)


@app.cell
def _(pred_fig):
    pred_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Feature Importance in California Housing""")
    return


@app.cell
def _(feature_names_housing, np, plt, rf_housing):
    importance_housing_fig, importance_housing_ax = plt.subplots(figsize=(10, 6))
    housing_importances = rf_housing.feature_importances_
    sorted_idx = np.argsort(housing_importances)
    importance_housing_ax.barh(np.array(feature_names_housing)[sorted_idx], housing_importances[sorted_idx])
    importance_housing_ax.set_xlabel("Importance")
    importance_housing_ax.set_title("Feature Importance in Random Forest (California Housing)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Key Takeaways

    1. **Random Forests are versatile** - They work well for both classification and regression tasks

    2. **Handle non-linear relationships** - As seen with make_moons, random forests can learn complex, non-linear decision boundaries without explicit feature engineering

    3. **Feature importance** - Random forests provide insight into which features are most important for predictions, helping with feature selection and model interpretation

    4. **Robustness** - By combining multiple trees trained on different data samples, random forests are less prone to overfitting than single decision trees

    5. **Real-world applicability** - The California housing example shows that random forests work effectively on real datasets with multiple features and continuous target values

    ### When to Use Random Forests:
    - Large datasets with many features
    - Non-linear relationships in data
    - Need for feature importance insights
    - When ensemble methods provide better generalization
    - Both classification and regression problems

    ### Hyperparameters to Tune:
    - `n_estimators`: Number of trees in the forest (higher = more robust but slower)
    - `max_depth`: Maximum depth of trees (lower = less overfitting)
    - `min_samples_split`: Minimum samples required to split a node
    - `min_samples_leaf`: Minimum samples required at a leaf node
    - `max_features`: Number of features to consider at each split
    """
    )
    return


if __name__ == "__main__":
    app.run()
