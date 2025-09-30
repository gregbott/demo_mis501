import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from datetime import datetime, date
    return mo, pl


@app.cell
def _(mo):
    mo.md(
        r"""
    # Polars Tutorial
    ## Introduction and Setup

    ```bash
    pixi add polars
    ```
    """
    )
    return


@app.cell
def _(pl):
    print(f"Polars version: {pl.__version__}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What is Polars?
    Polars is a blazingly fast DataFrame library for Python (and Rust).

    Key characteristics:

    - Built in Rust for maximum performance
    - Lazy evaluation with query optimization
    - Memory efficient with Apache Arrow backend
    - Expressive API similar to pandas but more consistent
    - Multi-threaded operations by default
    - No index-based operations (unlike pandas)

    Pandas is another DataFrame library released in 2008 by Wes McKinney. Polars is designed to handle larger-than-memory datasets efficiently and provides a more predictable performance profile than pandas.
    """
    )
    return


@app.cell
def _(pl):

    ## DataFrame vs Series


    ### Understanding DataFrame vs Series

    ### A Series is a 1-dimensional data structure (like a column)
    _series_example = pl.Series("temperatures", [20.5, 22.1, 19.8, 23.4, 21.0])
    print("Series example:")
    print(f"Series data type: {_series_example.dtype}")
    print(f"Series length: {len(_series_example)}")
    print(f"Object type: {type(_series_example)}")
    _series_example
    return


@app.cell
def _(pl):
    ### A DataFrame is a 2-dimensional data structure (like a table)
    df_example = pl.DataFrame({
        "city": ["New York", "London", "Tokyo", "Sydney", "Paris"],
        "temperature": [20.5, 22.1, 19.8, 23.4, 21.0],
        "humidity": [65, 70, 80, 55, 60]
    })


    print(f"DataFrame shape: {df_example.shape}")
    print(f"DataFrame columns: {df_example.columns}")
    print("DataFrame example:")
    df_example
    return (df_example,)


@app.cell
def _(df_example):
    df_example
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Polars Has No Index

    Key difference from pandas:

    - Polars DataFrames have NO INDEX
    - Rows are identified by their position (0, 1, 2, ...)
    - No .loc, .iloc, or index-based operations (however you can use .item(x, y))
    - This simplifies operations and improves performance
    - Use .with_row_index() if you need row numbers as a column
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Creating DataFrames

    ## From dictionaries
    """
    )
    return


@app.cell
def _(pl):
    _dict_data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "salary": [50000, 60000, 70000, 55000],
        "department": ["Engineering", "Marketing", "Engineering", "Sales"]
    }

    _df_from_dict = pl.DataFrame(_dict_data)
    _df_from_dict
    return


@app.cell
def _(mo):
    mo.md(r"""## From Lists""")
    return


@app.cell
def _(pl):
    _list_data = [
        ["Product A", 100, 25.99],
        ["Product B", 150, 19.99],
        ["Product C", 75, 35.50],
        ["Product D", 200, 15.99]
    ]

    _df_from_lists = pl.DataFrame(
        _list_data,
        schema=["product_name", "quantity", "price"],
        orient="row"
    )

    _df_from_lists
    return


@app.cell
def _(mo):
    mo.md(r"""## From files (csv, xlsx, json, parquet,...)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Schema control

    ### dtypes


    ### schema_overrides
    """
    )
    return


@app.cell
def _(pl):
    # df = pl.read_csv('../data/stores_sales_forecasting.csv')
    df = (pl.read_csv('../data/stores_sales_forecasting2.csv',
                    encoding='cp1252',
                    schema_overrides={
                            'Row ID':pl.UInt16,
                            'Postal Code':pl.String,
                            'Quantity':pl.UInt16,
                            }
                     )
     
         )
    df.shape
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Inspecting data
    ## Initial exploration

    ### head(), tail(), describe(), schema

    """
    )
    return


@app.cell
def _(df):
    dict(df.schema) # wrapped in dict() constructor so Marimo prints vertically
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### glimpse()

    ```python
    glimpse()
    ```
    Returns a formatted string showing a summary of the DataFrame including:

    - Number of rows and columns
    - Column names
    - data types
    - a preview of the first few values
    """
    )
    return


@app.cell
def _(df):
    # df.item(1,2) = 'November 8, 2016' # Not allowed in Polars, DataFrames are immutable
    df.item(1,2)
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _():
    return


@app.cell
def _(df, pl):
    # Keep original string column while creating parsed version
    df_sales = df.with_columns([
        pl.col("Order Date")
        .str.strptime(pl.Date, format="%m/%d/%Y", strict=False)
        .alias("Order Date Parsed")
    ])

    # Filter for nulls in parsed column (these are failures)
    parse_failures = df_sales.filter(
        pl.col("Order Date Parsed").is_null() & 
        pl.col("Order Date").is_not_null()  # Original had a value
    )

    print(f"Rows that failed to parse: {len(parse_failures)}")
    print(parse_failures.select(["Order Date", "Order Date Parsed"]))
    return


@app.cell
def _():
    # Eager
    # iowa_df1 = pl.read_csv('../data/Iowa_Liquor_Sales-26M.csv.gz', infer_schema_length=100000)
    # iowa_df1 = pl.read_csv('../data/Iowa_Liquor_Sales-26M.csv.gz', schema_overrides={'Zip Code':pl.String, 'Item Number':pl.String})
    return


@app.cell
def _(lazy_sample):
    dict(lazy_sample.schema)
    return


@app.cell
def _():


    # iowa_df = pl.scan_csv('../data/Iowa_Liquor_Sales-26M.csv.gz')
    # result = iowa_df.filter
    return


@app.cell
def _(pl):
    ## Basic DataFrame Methods - Inspection


    # Basic inspection methods using our sample data
    sample_df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 60000, 70000, 55000, 65000],
        "department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"]
    })

    print("Original DataFrame:")
    print(sample_df)

    print(f"\nShape: {sample_df.shape}")
    print(f"Columns: {sample_df.columns}")
    print(f"Data types:\n{sample_df.dtypes}")

    print("\nFirst 3 rows:")
    print(sample_df.head(3))

    print("\nLast 2 rows:")
    print(sample_df.tail(2))
    return (sample_df,)


@app.cell
def _(sample_df):
    ## Basic DataFrame Methods - Summary Statistics


    # Summary statistics and info
    print("Summary statistics:")
    print(sample_df.describe())

    print("\n" + "="*50 + "\n")

    print("Info about the DataFrame:")
    print(sample_df.schema)

    print("\n" + "="*50 + "\n")

    print("Null value counts:")
    print(sample_df.null_count())

    print("\nUnique value counts for 'department' column:")
    print(sample_df["department"].value_counts())
    return


@app.cell
def _(pl, sample_df):
    ## Basic DataFrame Methods - Selection and Filtering


    # Column selection
    print("Select single column (returns DataFrame):")
    _single_col_df = sample_df.select("name")
    print(_single_col_df)

    print("\nSelect single column (returns Series):")
    _single_col_series = sample_df["name"]
    print(_single_col_series)

    print("\nSelect multiple columns:")
    _multi_cols = sample_df.select(["name", "age", "salary"])
    print(_multi_cols)

    print("\n" + "="*40 + "\n")

    # Row filtering
    print("Filter rows where age > 30:")
    _filtered_df = sample_df.filter(pl.col("age") > 30)
    print(_filtered_df)

    print("\nFilter rows with multiple conditions:")
    _complex_filter = sample_df.filter(
        (pl.col("age") > 25) & (pl.col("department") == "Engineering")
    )
    print(_complex_filter)
    return


@app.cell
def _(pl, sample_df):
    ## Basic DataFrame Methods - Sorting and Grouping


    # Sorting
    print("Sort by age (ascending):")
    _sorted_asc = sample_df.sort("age")
    print(_sorted_asc)

    print("\nSort by salary (descending):")
    _sorted_desc = sample_df.sort("salary", descending=True)
    print(_sorted_desc)

    print("\nSort by multiple columns:")
    _multi_sort = sample_df.sort(["department", "age"])
    print(_multi_sort)

    print("\n" + "="*40 + "\n")

    # Grouping and aggregation
    print("Group by department and calculate mean salary:")
    _grouped = sample_df.group_by("department").agg(
        pl.col("salary").mean().alias("avg_salary"),
        pl.col("age").mean().alias("avg_age"),
        pl.len().alias("count")
    )
    print(_grouped)
    return


@app.cell
def _(pl, sample_df):
    ## Basic DataFrame Methods - Adding and Modifying Columns


    # Adding new columns
    print("Add new columns:")
    _df_with_new_cols = sample_df.with_columns([
        (pl.col("salary") * 0.1).alias("bonus"),
        pl.when(pl.col("age") >= 30)
        .then(pl.lit("Senior"))
        .otherwise(pl.lit("Junior"))
        .alias("level")
    ])

    print(_df_with_new_cols)

    print("\n" + "="*40 + "\n")

    # Renaming columns
    print("Rename columns:")
    _renamed_df = sample_df.rename({
        "name": "employee_name",
        "department": "dept"
    })
    print(_renamed_df)

    print("\nDrop columns:")
    _dropped_cols = sample_df.drop(["id", "department"])
    print(_dropped_cols)
    return


@app.cell
def _(pl, sample_df):
    ## Method Chaining Example


    # Polars supports method chaining for complex operations
    print("Method chaining example:")
    print("Filter → Group → Aggregate → Sort")

    _chained_result = (
        sample_df
        .filter(pl.col("age") >= 28)  # Filter adults
        .group_by("department")       # Group by department
        .agg([                        # Multiple aggregations
            pl.col("salary").mean().alias("avg_salary"),
            pl.col("age").max().alias("max_age"),
            pl.len().alias("employee_count")
        ])
        .sort("avg_salary", descending=True)  # Sort by average salary
    )

    print(_chained_result)
    return


@app.cell
def _():
    ## Key Takeaways


    _takeaways = """
    Key Polars Concepts Summary:

    1. DataFrame: 2D table structure (rows & columns)
    2. Series: 1D column structure  
    3. No Index: Rows identified by position only
    4. Lazy Evaluation: Use scan_* functions for large datasets
    5. Expressions: Use pl.col() for column references
    6. Method Chaining: Operations can be chained together
    7. Performance: Built for speed with Rust backend

    Common patterns:
    - df.select() for column selection
    - df.filter() for row filtering  
    - df.with_columns() for adding/modifying columns
    - df.group_by().agg() for aggregations
    - Chain methods together for complex operations

    Next steps:
    - Learn about expressions and lazy evaluation
    - Explore advanced aggregations and window functions
    - Practice with real datasets
    """

    print(_takeaways)
    return


@app.cell
def _(pl):
    ## Practice Exercise


    # Practice exercise: Create and manipulate a DataFrame
    print("Practice Exercise: Sales Data Analysis")

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
    return


if __name__ == "__main__":
    app.run()
