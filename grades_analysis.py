import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return mo,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Student Grades Analysis

        This notebook demonstrates how to work with Excel data in Python using pandas.
        We'll explore:
        - **Horizontal calculations**: Computing values across columns (e.g., average grade per student)
        - **Vertical calculations**: Computing values down rows (e.g., average per subject)
        - **Pivot tables**: Reshaping and summarizing data
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""## 1. Loading the Data""")
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np

    # Load the Excel file
    df = pd.read_excel('data/grades.xlsx')
    df
    return df, np, pd


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## 2. Horizontal Calculations

        Horizontal calculations work **across columns** for each row.
        This is useful when you want to aggregate data for individual records (e.g., calculating total or average scores for each student).
        """
    )
    return


@app.cell
def __(df, pd):
    # Calculate the average grade for each student across all subjects
    # We select only the numeric columns (Math, Science, English, History)
    subject_columns = ['Math', 'Science', 'English', 'History']

    # Create a copy with the average grade column
    df_with_avg = df.copy()
    df_with_avg['Average'] = df_with_avg[subject_columns].mean(axis=1)  # axis=1 means across columns
    df_with_avg['Total'] = df_with_avg[subject_columns].sum(axis=1)

    # Round to 2 decimal places for better display
    df_with_avg['Average'] = df_with_avg['Average'].round(2)

    df_with_avg
    return df_with_avg, subject_columns


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        **Key concept**: `axis=1` tells pandas to perform the operation across columns (horizontally).

        - `mean(axis=1)`: Calculate mean across columns for each row
        - `sum(axis=1)`: Calculate sum across columns for each row
        - `max(axis=1)`: Find maximum value across columns for each row
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### More Horizontal Calculations""")
    return


@app.cell
def __(df, subject_columns):
    # Additional horizontal calculations
    horizontal_stats = df.copy()
    horizontal_stats['Average'] = df[subject_columns].mean(axis=1).round(2)
    horizontal_stats['Min_Grade'] = df[subject_columns].min(axis=1)
    horizontal_stats['Max_Grade'] = df[subject_columns].max(axis=1)
    horizontal_stats['Range'] = horizontal_stats['Max_Grade'] - horizontal_stats['Min_Grade']

    horizontal_stats
    return horizontal_stats,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## 3. Vertical Calculations

        Vertical calculations work **down rows** for each column.
        This is useful when you want to aggregate data across all records (e.g., calculating class average per subject).
        """
    )
    return


@app.cell
def __(df, subject_columns):
    # Calculate statistics for each subject across all students
    vertical_stats = df[subject_columns].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    vertical_stats
    return vertical_stats,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        **Key concept**: `axis=0` (default) tells pandas to perform the operation down rows (vertically).

        - `mean(axis=0)` or `mean()`: Calculate mean down rows for each column
        - `sum()`: Calculate sum down rows for each column
        - `agg([...])`: Apply multiple aggregation functions at once
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Class Performance Summary""")
    return


@app.cell
def __(df, pd, subject_columns):
    # Create a summary showing vertical calculations
    summary_data = {
        'Subject': subject_columns,
        'Class Average': [df[col].mean().round(2) for col in subject_columns],
        'Highest Score': [df[col].max() for col in subject_columns],
        'Lowest Score': [df[col].min() for col in subject_columns],
        'Std Deviation': [df[col].std().round(2) for col in subject_columns]
    }

    class_summary = pd.DataFrame(summary_data)
    class_summary
    return class_summary, summary_data


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## 4. Pivot Tables

        Pivot tables are powerful tools for reshaping and summarizing data. They allow you to:
        - Group data by categories
        - Apply aggregation functions
        - Create cross-tabulations
        - Analyze data from different perspectives
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Example 1: Average Grades by Class""")
    return


@app.cell
def __(df, pd, subject_columns):
    # Pivot table showing average grades for each class
    pivot1 = pd.pivot_table(
        df,
        values=subject_columns,  # Columns to aggregate
        index='Class',            # Rows of the pivot table
        aggfunc='mean'            # Aggregation function
    ).round(2)

    pivot1
    return pivot1,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        This pivot table shows the average score for each subject, grouped by class.
        We can quickly see how Class A and Class B compare across different subjects.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Example 2: Multiple Aggregations""")
    return


@app.cell
def __(df, pd):
    # Pivot table with multiple aggregation functions
    pivot2 = pd.pivot_table(
        df,
        values='Math',           # We'll focus on Math scores
        index='Class',
        aggfunc=['mean', 'min', 'max', 'count']
    ).round(2)

    pivot2
    return pivot2,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        This pivot table shows multiple statistics for Math scores by class:
        - **mean**: Average Math score per class
        - **min**: Lowest Math score per class
        - **max**: Highest Math score per class
        - **count**: Number of students per class
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Example 3: Custom Pivot with Performance Categories""")
    return


@app.cell
def __(df, pd, subject_columns):
    # Create a performance category based on average grade
    df_enhanced = df.copy()
    df_enhanced['Average'] = df_enhanced[subject_columns].mean(axis=1)

    # Categorize students
    def categorize_performance(avg):
        if avg >= 90:
            return 'Excellent'
        elif avg >= 80:
            return 'Good'
        elif avg >= 70:
            return 'Satisfactory'
        else:
            return 'Needs Improvement'

    df_enhanced['Performance'] = df_enhanced['Average'].apply(categorize_performance)

    df_enhanced
    return categorize_performance, df_enhanced


@app.cell
def __(df_enhanced, pd):
    # Pivot table: Count students by Class and Performance category
    pivot3 = pd.pivot_table(
        df_enhanced,
        values='Student',        # We'll count students
        index='Class',
        columns='Performance',   # Create columns for each performance category
        aggfunc='count',
        fill_value=0             # Fill missing combinations with 0
    )

    pivot3
    return pivot3,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        This cross-tabulation shows how many students in each class fall into different performance categories.
        This type of pivot table is useful for understanding the distribution of performance across groups.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Example 4: Advanced Pivot - Subject Averages by Performance Category""")
    return


@app.cell
def __(df_enhanced, pd, subject_columns):
    # Reshape data to long format for more complex pivoting
    df_long = pd.melt(
        df_enhanced,
        id_vars=['Student', 'Class', 'Performance'],
        value_vars=subject_columns,
        var_name='Subject',
        value_name='Score'
    )

    # Pivot: Average score by Performance category and Subject
    pivot4 = pd.pivot_table(
        df_long,
        values='Score',
        index='Performance',
        columns='Subject',
        aggfunc='mean'
    ).round(2)

    pivot4
    return df_long, pivot4


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        This pivot table shows the average score for each subject, broken down by performance category.
        It helps identify if certain performance categories struggle with specific subjects.

        **Note**: We used `pd.melt()` to transform the data from wide to long format, making it easier to create this type of pivot.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Summary

        ### Key Takeaways:

        **Horizontal Calculations (axis=1)**:
        - Work across columns for each row
        - Example: `df.mean(axis=1)` calculates row-wise means
        - Use for per-record aggregations

        **Vertical Calculations (axis=0 or default)**:
        - Work down rows for each column
        - Example: `df.mean()` calculates column-wise means
        - Use for overall statistics per feature

        **Pivot Tables**:
        - `pd.pivot_table()` creates summary tables
        - Parameters:
          - `values`: columns to aggregate
          - `index`: rows of pivot table
          - `columns`: columns of pivot table
          - `aggfunc`: aggregation function(s)
        - Use `pd.melt()` to reshape data when needed
        """
    )
    return


if __name__ == "__main__":
    app.run()
