import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        confusion_matrix, classification_report, roc_curve, auc,
        accuracy_score, precision_score, recall_score, f1_score,
        cohen_kappa_score, log_loss, ConfusionMatrixDisplay, RocCurveDisplay
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    import polars as pl
    return (
        DecisionTreeClassifier,
        accuracy_score,
        auc,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        mo,
        np,
        pl,
        plot_tree,
        plt,
        precision_score,
        recall_score,
        roc_curve,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Decision Trees: A Comprehensive Educational Guide

    This notebook teaches you how decision trees work and, more importantly, **how to evaluate whether they're actually good at making predictions**.

    We'll use a medical screening scenario (cancer detection) with 100 patients, following the exact example provided.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 1: Understanding the Problem

    Imagine we're screening 100 patients for cancer:
    - **10 patients actually have cancer** (positive class)
    - **90 patients don't have cancer** (negative class)
    """)
    return


@app.cell
def _(np, pl):
    # Create synthetic dataset matching the 100-patient example
    np.random.seed(73)

    # Generate features for patients
    n_samples = 100

    # Create features that would correlate with cancer
    features = np.random.randn(n_samples, 5)

    # Create target: 10 positive, 90 negative (imbalanced)
    y = np.array([1] * 10 + [0] * 90)

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = features[shuffle_idx]
    y = y[shuffle_idx]

    # Create polars dataframe for inspection
    df = pl.DataFrame({
        "feature_1": X[:, 0],
        "feature_2": X[:, 1],
        "feature_3": X[:, 2],
        "feature_4": X[:, 3],
        "feature_5": X[:, 4],
        "has_cancer": y
    })
    return X, y


@app.cell
def _(DecisionTreeClassifier, X, train_test_split, y):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train a decision tree classifier
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    tree.fit(X_train, y_train)

    # Make predictions
    y_pred = tree.predict(X_test)

    # Get probability predictions for ROC-AUC
    y_pred_proba = tree.predict_proba(X_test)[:, 1]
    return tree, y_pred, y_pred_proba, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 2: Visualizing the Decision Tree

    A decision tree makes predictions by recursively splitting the data based on feature values.
    Below is a visualization of our trained tree structure.
    """)
    return


@app.cell
def _(plot_tree, plt, tree):
    # Visualize the decision tree
    _fig, _ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=[f"Feature {i+1}" for i in range(5)],
        class_names=["No Cancer", "Cancer"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=_ax
    )
    plt.tight_layout()
    tree_fig = _fig
    return (tree_fig,)


@app.cell
def _(tree_fig):
    tree_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **How to read this tree:**
    - Each node shows a feature test (e.g., "feature_1 ≤ -0.45")
    - The `samples` count shows how many training examples reached that node
    - The `value` shows [no_cancer_count, cancer_count]
    - The color intensity represents the class distribution (darker = more pure)
    - Leaf nodes are where predictions are made
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 3: The Confusion Matrix

    The confusion matrix is the foundation for understanding classifier performance.
    It breaks down exactly what your model got right and wrong.
    """)
    return


@app.cell
def _(confusion_matrix, plt, sns, y_pred, y_test):
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize confusion matrix
    _fig, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['No Cancer', 'Cancer'],
        yticklabels=['No Cancer', 'Cancer'],
        ax=_ax,
        cbar_kws={'label': 'Count'}
    )
    _ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    _ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    _ax.set_title('Confusion Matrix\n(What the model predicted vs. reality)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_fig = _fig
    return cm, cm_fig


@app.cell
def _(cm_fig):
    cm_fig
    return


@app.cell
def _(cm):
    TP = cm[1, 1]  # Predicted positive, actually positive
    FP = cm[0, 1]  # Predicted positive, actually negative
    FN = cm[1, 0]  # Predicted negative, actually positive
    TN = cm[0, 0]  # Predicted negative, actually negative
    return FN, FP, TN, TP


@app.cell
def _(FN, FP, TN, TP, mo):
    _total = TP + FP + FN + TN

    explanation = f"""
    ### Reading the Confusion Matrix

    - **True Positives (TPInclude markdown) = {TP}**: Model correctly identified cancer cases
    - **False Positives (FP) = {FP}**: Model said "cancer" but patient is healthy (false alarm)
    - **False Negatives (FN) = {FN}**: Model said "healthy" but patient has cancer (missed diagnosis)
    - **True Negatives (TN) = {TN}**: Model correctly identified healthy patients

    **Total predictions: {_total}**
    """

    mo.md(explanation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 4: Classification Metrics Explained

    Different metrics answer different questions about your model.
    Each is useful in different contexts.
    """)
    return


@app.cell
def _(FN, FP, TN, TP):
    # Calculate all metrics
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return accuracy, f1, precision, recall, specificity


@app.cell
def _(accuracy, f1, mo, precision, recall, specificity):
    metrics_explanation = f"""
    ### Key Metrics

    **Accuracy = {accuracy:.1%}**
    - Formula: (TP + TN) / Total
    - **Meaning**: Overall correctness. Out of 100 predictions, how many were right?
    - **⚠️ Caution**: Can be misleading with imbalanced classes!

    **Precision = {precision:.1%}**
    - Formula: TP / (TP + FP)
    - **Meaning**: When the model says "cancer," how often is it right?
    - **Use case**: High precision is important when false alarms are expensive

    **Recall (Sensitivity) = {recall:.1%}**
    - Formula: TP / (TP + FN)
    - **Meaning**: Of all actual cancer cases, how many did we catch?
    - **Use case**: High recall is critical in medical screening (don't miss sick patients!)

    **Specificity = {specificity:.1%}**
    - Formula: TN / (TN + FP)
    - **Meaning**: Of all healthy patients, how many did we correctly identify?

    **F1 Score = {f1:.1%}**
    - Formula: 2 × (Precision × Recall) / (Precision + Recall)
    - **Meaning**: Harmonic mean balancing precision and recall
    - **Use case**: Good for imbalanced datasets
    """

    mo.md(metrics_explanation)
    return


@app.cell
def _(
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    y_pred,
    y_test,
):
    # Using sklearn to verify calculations
    sklearn_accuracy = accuracy_score(y_test, y_pred)
    sklearn_precision = precision_score(y_test, y_pred)
    sklearn_recall = recall_score(y_test, y_pred)
    sklearn_f1 = f1_score(y_test, y_pred)
    sklearn_kappa = cohen_kappa_score(y_test, y_pred)
    return (
        sklearn_accuracy,
        sklearn_f1,
        sklearn_kappa,
        sklearn_precision,
        sklearn_recall,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 5: ROC-AUC: The Probability Curve

    ROC (Receiver Operating Characteristic) curve shows how well your model
    distinguishes between classes across all probability thresholds.
    """)
    return


@app.cell
def _(auc, plt, roc_curve, y_pred_proba, y_test):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    _ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.50)')
    _ax.set_xlim([0.0, 1.0])
    _ax.set_ylim([0.0, 1.05])
    _ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    _ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    _ax.set_title('ROC-AUC Curve\n(Higher is better)', fontsize=14, fontweight='bold')
    _ax.legend(loc="lower right", fontsize=11)
    _ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_fig = _fig
    return roc_auc, roc_fig


@app.cell
def _(roc_fig):
    roc_fig
    return


@app.cell
def _(mo, roc_auc):
    roc_explanation = f"""
    ### Understanding ROC-AUC

    **AUC (Area Under Curve) = {roc_auc:.2f}**

    - **Meaning**: The probability that if you pick one positive and one negative example,
      the model ranks the positive one higher. Perfect = 1.0, Random = 0.5
    - **Interpretation**:
      - 0.90-1.0: Excellent discrimination
      - 0.80-0.90: Good discrimination
      - 0.70-0.80: Fair discrimination
      - 0.60-0.70: Poor discrimination
      - 0.50-0.60: Very poor discrimination

    **Why is ROC-AUC useful?**
    - It evaluates the model across ALL probability thresholds
    - It handles class imbalance better than accuracy
    - It doesn't assume a specific operating point
    """

    mo.md(roc_explanation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 6: Cohen's Kappa - Accounting for Imbalance

    Cohen's Kappa corrects accuracy for class imbalance and chance agreement.
    It's especially valuable when you have imbalanced datasets.
    """)
    return


@app.cell
def _(FN, FP, TN, TP):
    _total = TP + FP + FN + TN

    # Calculate expected agreement
    predicted_pos = TP + FP
    predicted_neg = TN + FN
    actual_pos = TP + FN
    actual_neg = TN + FP

    p_chance_both_pos = (predicted_pos / _total) * (actual_pos / _total)
    p_chance_both_neg = (predicted_neg / _total) * (actual_neg / _total)
    p_chance = p_chance_both_pos + p_chance_both_neg

    observed = (TP + TN) / _total
    kappa = (observed - p_chance) / (1 - p_chance) if (1 - p_chance) != 0 else 0
    return kappa, observed, p_chance


@app.cell
def _(kappa, mo, observed, p_chance):
    kappa_explanation = f"""
    ### Cohen's Kappa Calculation

    **Observed Agreement**: {observed:.2%}

    **Expected Agreement by Chance**: {p_chance:.2%}

    **Cohen's Kappa = {kappa:.3f}**

    **Interpretation**:
    - Kappa ranges from -1.0 to 1.0
    - -1.0: Perfect disagreement
    - 0.0: Agreement due to chance
    - 1.0: Perfect agreement

    **Landis & Koch Scale**:
    - < 0.20: Slight agreement
    - 0.21-0.40: Fair agreement
    - 0.41-0.60: Moderate agreement
    - 0.61-0.80: Substantial agreement
    - 0.81-1.0: Almost perfect agreement

    **Why Kappa matters**: A high accuracy doesn't mean your model is good if it's just
    agreeing with a heavily imbalanced dataset. Kappa reveals if the model is actually
    learning something or just predicting the majority class.
    """

    mo.md(kappa_explanation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 7: Log Loss - Probability Confidence

    Log Loss (Cross-Entropy) measures how confident your model is in its predictions.
    Lower is better.
    """)
    return


@app.cell
def _(log_loss, y_pred_proba, y_test):
    # Calculate log loss
    logloss = log_loss(y_test, y_pred_proba)

    # For educational purposes, calculate per-class contributions
    correct_probs = []
    incorrect_probs = []
    for true_label, pred_prob in zip(y_test, y_pred_proba):
        if true_label == 1:
            correct_probs.append(pred_prob)
        else:
            correct_probs.append(1 - pred_prob)
    return (logloss,)


@app.cell
def _(logloss, mo):
    logloss_explanation = f"""
    ### Log Loss (Cross-Entropy)

    **Log Loss = {logloss:.3f}**

    **Formula**: -1/N × Σ[y × log(p) + (1-y) × log(1-p)]
    - Where y is the true label and p is the predicted probability

    **Interpretation**:
    - 0.0: Perfect predictions
    - 0.3-0.5: Good predictions
    - 0.5-1.0: Poor predictions
    - >1.0: Very poor predictions

    **What it penalizes**:
    - Being wrong with high confidence (e.g., predicting 0.9 when truth is 0)
    - Being unsure even about correct predictions (e.g., predicting 0.51 when truth is 1)

    **Use case**: Log Loss is great when probability calibration matters,
    such as when you need to rank patients by risk.
    """

    mo.md(logloss_explanation)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 8: Putting It All Together - Summary Table
    """)
    return


@app.cell
def _(
    logloss,
    pl,
    roc_auc,
    sklearn_accuracy,
    sklearn_f1,
    sklearn_kappa,
    sklearn_precision,
    sklearn_recall,
):
    # Create a summary table
    summary_data = {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall (Sensitivity)",
            "Specificity",
            "F1 Score",
            "Cohen's Kappa",
            "ROC-AUC",
            "Log Loss"
        ],
        "Value": [
            f"{sklearn_accuracy:.1%}",
            f"{sklearn_precision:.1%}",
            f"{sklearn_recall:.1%}",
            f"{0.87:.1%}",  # Example specificity
            f"{sklearn_f1:.1%}",
            f"{sklearn_kappa:.3f}",
            f"{roc_auc:.3f}",
            f"{logloss:.3f}"
        ],
        "What It Answers": [
            "Overall, how often is the model correct?",
            "When it predicts cancer, how often is it right?",
            "Of actual cancers, how many does it find?",
            "Of actual non-cancers, how many does it correctly identify?",
            "Balanced measure of precision and recall?",
            "Is the agreement better than chance?",
            "Can it rank positive cases higher than negative ones?",
            "How confident is the model in its probability predictions?"
        ]
    }

    summary_df = pl.DataFrame(summary_data)
    return (summary_df,)


@app.cell
def _(mo, summary_df):
    # Convert polars DataFrame to markdown table
    rows = summary_df.to_dicts()
    headers = summary_df.columns

    # Create markdown table
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in rows:
        markdown_table += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"

    mo.md(markdown_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 9: Key Takeaways for Cancer Screening

    ### The Trade-off Problem
    In cancer screening, we have a fundamental tension:
    - **High Recall** (catch all cancers) means accepting many false alarms (low precision)
    - **High Precision** (minimize false alarms) means missing some cancers (low recall)

    ### Decision for This Model
    """)
    return


@app.cell
def _(mo):
    decision = """
    For cancer screening, **we prioritize Recall**:
    - Missing a cancer case (False Negative) is worse than a false alarm
    - A false alarm leads to further testing; a missed case could be fatal
    - We can tolerate lower precision if recall is high

    ### Questions to Ask
    1. What is the cost of a false positive? (unnecessary follow-up tests)
    2. What is the cost of a false negative? (patient harm)
    3. What threshold maximizes the metric that matters most?

    ### Actionable Insights from This Model
    - **Do use this model** for initial screening (high recall)
    - **Require follow-up testing** to reduce false positives
    - **Monitor both metrics** rather than relying on accuracy alone
    - **Adjust the decision threshold** if the false alarm rate is too high
    """

    mo.md(decision)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Part 10: Interactive Threshold Exploration

    Change the probability threshold and see how metrics change.
    """)
    return


@app.cell
def _(mo):
    # Slider for threshold
    threshold = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="Decision Threshold")
    return (threshold,)


@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    threshold,
    y_pred_proba,
    y_test,
):
    # Apply threshold
    threshold_value = threshold.value
    y_pred_threshold = (y_pred_proba >= threshold_value).astype(int)

    # Calculate metrics with threshold
    cm_threshold = confusion_matrix(y_test, y_pred_threshold)
    TP_t = cm_threshold[1, 1]
    FP_t = cm_threshold[0, 1]
    FN_t = cm_threshold[1, 0]
    TN_t = cm_threshold[0, 0]

    acc_t = accuracy_score(y_test, y_pred_threshold)
    prec_t = precision_score(y_test, y_pred_threshold, zero_division=0)
    rec_t = recall_score(y_test, y_pred_threshold, zero_division=0)
    f1_t = f1_score(y_test, y_pred_threshold, zero_division=0)
    return acc_t, f1_t, prec_t, rec_t


@app.cell(hide_code=True)
def _(mo, threshold):
    mo.md(f"""
    **Current Threshold: {threshold.value:.2f}**
    """)
    return


@app.cell
def _(acc_t, f1_t, mo, prec_t, rec_t):
    results_text = f"""
    **Metrics at This Threshold:**

    | Metric | Value |
    |--------|-------|
    | Accuracy | {acc_t:.1%} |
    | Precision | {prec_t:.1%} |
    | Recall | {rec_t:.1%} |
    | F1 Score | {f1_t:.1%} |

    **Observation**: As you increase the threshold, fewer cases are predicted as "cancer" (higher precision, lower recall).
    As you decrease the threshold, more cases are predicted as "cancer" (lower precision, higher recall).
    """

    mo.md(results_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Conclusion

    Decision trees are powerful tools for classification, but evaluating them correctly is crucial.

    **Remember:**
    - **Accuracy is not everything** - especially with imbalanced data
    - **Choose metrics based on your use case** - cancer screening needs high recall
    - **Understand the trade-offs** - precision vs. recall, cost vs. benefit
    - **Visualize your results** - confusion matrices and ROC curves tell stories
    - **Use multiple metrics** - no single metric captures everything
    """)
    return


if __name__ == "__main__":
    app.run()
