import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _(mo):
    mo.md(r"""# AN - Basic DataFrame Operations""")
    return


@app.cell
def _(pl):
    # Ingest the grades.xlsx file and store in grades_df
    grades_df = pl.read_excel('data/grades.xlsx')

    # Show a list of columns, their data types, and the first few values
    grades_df.glimpse() # glimpse() is the best answer here, take off if they did something more cumbersome
    return (grades_df,)


@app.cell
def _(grades_df):
    # How many students are there? Write code that gets the count of students.

    grades_df['last_name'].count() # Accept any programmatic way they came up with the correct total.
    return


@app.cell
def _(grades_df):
    # Show a count of missing values, if any.
    grades_df.null_count() # value_counts

    # Which columns have missing values? How many per column?
    #     Answer: No columns have missing values.
    return


@app.cell
def _(grades_df, pl):
    # Add a quiz average column

    grades_avg_df = grades_df.with_columns([
        pl.mean_horizontal(['quiz_1', 'quiz_2', 'quiz_3', 'quiz_4']).round(2).alias('quiz_avg'),
        pl.mean_horizontal(['essay_1', 'essay_2', 'essay_3']).round(2).alias('essay_avg'),
        pl.mean_horizontal(['midterm_exam', 'final_exam']).round(2).alias('exam_avg'),
    
        (pl.mean_horizontal(['quiz_1', 'quiz_2', 'quiz_3', 'quiz_4']) * 0.20 +
         pl.mean_horizontal(['essay_1', 'essay_2', 'essay_3']) * 0.30 +
         pl.mean_horizontal(['midterm_exam', 'final_exam']) * 0.50).round(2).alias('final_grade')
    ])
    grades_avg_df
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Practice exercise: Create and manipulate a DataFrame
    print("Practice Exercise: Sales Data Analysis")

    # Ingest the sales CSV file


    # 

    _sales_data = pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "product": ["Laptop", "Mouse", "Laptop", "Keyboard", "Mouse"],
        "quantity": [2, 10, 1, 5, 8],
        "unit_price": [1000, 25, 1000, 75, 25],
        "sales_rep": ["John", "Mary", "John", "Bob", "Mary"]
    })

    print("Sales data:")
    print(_sales_data)

    # Try these exercises:
    print("\nExercise solutions:")

    # 1. Add a total_amount column
    _exercise_1 = _sales_data.with_columns(
        (pl.col("quantity") * pl.col("unit_price")).alias("total_amount")
    )
    print("\n1. DataFrame with total_amount column:")
    print(_exercise_1)

    # 2. Find total sales by sales rep
    _exercise_2 = _exercise_1.group_by("sales_rep").agg(
        pl.col("total_amount").sum().alias("total_sales")
    ).sort("total_sales", descending=True)
    print("\n2. Total sales by sales rep:")
    print(_exercise_2)

    # 3. Filter for high-value sales (>= $1000)
    _exercise_3 = _exercise_1.filter(pl.col("total_amount") >= 1000)
    print("\n3. High-value sales (>= $1000):")
    print(_exercise_3)
    """
    )
    return


if __name__ == "__main__":
    app.run()
