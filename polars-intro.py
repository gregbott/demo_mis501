import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from datetime import datetime, date
    import pathlib
    import chardet

    data_path = pathlib.Path('/mnt/expansion16TB/Dropbox/3_Resources/marimo_data')
    return chardet, mo, pathlib, pl


@app.cell
def _(mo):
    mo.md(
        r"""
    # 1. Foundations & Philosophy

    **Why Polars?** Performance, memory efficiency, expressive API

    **Core concepts:** Immutability, lazy vs eager execution, expressions

    **Key differences from pandas:** No index, column-oriented, query optimization
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
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
    # 2. Loading & Inspecting Data

    Reading files: `read_csv()`, `read_parquet()`, `read_excel()`

    Schema control: `dtypes`, `schema_overrides`

    Initial exploration: `head()`, `tail()`, `glimpse()`, `describe()`, `schema`
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
    ## Schema control and Encoding

    ### dtypes


    ### schema_overrides
    """
    )
    return


@app.cell
def _(df):
    df.estimated_size()
    return


@app.cell
def _(pl):
    # When possible, have a good idea of the data you're ingesting. How many columns? How many rows? What fields? Types? Etc.

    # Error: First attempt. In its original format, the csv file was encoded using 'cp1252' but is read using a different encoding.
    #      No exception is raised, but it is misread. Note ZERO rows and 42,441 columns.

    sales_file_path = '/mnt/expansion16TB/Dropbox/3_Resources/marimo_data/stores_sales_forecasting.csv'

    # df = pl.read_csv(sales_file_path) # First attempt

    # Second attempt
    df = (pl.read_csv(sales_file_path,
                    encoding='cp1252',
                    schema_overrides={
                            'Row ID':pl.UInt16,
                            'Postal Code':pl.String,
                            'Quantity':pl.UInt16,
                            'Sales':pl.Float32,
                            'Profit':pl.Float32,
                            'Discount':pl.Float32
                            }
                     ).drop(['Country', 'Row ID'])
         )

    df.shape
    return df, sales_file_path


@app.cell
def _(chardet, pathlib, sales_file_path):
    with open(pathlib.Path(sales_file_path), 'rb') as file:
        # Read first 100,000 bytes or entire file if smaller
        raw_data = file.read(100000)

    # Detect encoding
    results = chardet.detect(raw_data)
    results
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df['Segment'].value_counts()
    return


@app.cell
def _(df):
    # If you see only types and no column names, you have a problem. In this case, it's an encoding error.
    dict(df.schema) # wrapped in dict() constructor so Marimo prints vertically
    return


@app.cell(hide_code=True)
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
def _(mo):
    mo.md(r"""### Return a single item""")
    return


@app.cell
def _(df):
    # df.item(1,2) = 'November 8, 2016' # Not allowed in Polars, DataFrames are immutable
    df.item(1,2)
    return


@app.cell
def _(mo):
    mo.md(r"""### Check Unique values""")
    return


@app.cell
def _(sample_df):
    sample_df["name"].value_counts()
    return


@app.cell
def _(df):
    df['State'].value_counts(sort=True, parallel=True, name="count")
    return


@app.cell
def _(df):
    df['State'].unique().sort()
    return


@app.cell
def _(df, pl):
    df.filter(pl.col('Postal Code').str.len_chars()<5)['Postal Code'].unique().sort()
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _():
    # Eager
    # iowa_df1 = pl.read_csv('../data/Iowa_Liquor_Sales-26M.csv.gz', infer_schema_length=100000)
    # iowa_df1 = pl.read_csv('../data/Iowa_Liquor_Sales-26M.csv.gz', schema_overrides={'Zip Code':pl.String, 'Item Number':pl.String})
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
def _(mo):
    mo.md(r"""## Summary Statistics""")
    return


@app.cell
def _(sample_df):
    sample_df.describe()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(sample_df):
    sample_df.schema
    return


@app.cell
def _(sample_df):
    sample_df.null_count()
    return


@app.cell
def _(sample_df):
    sample_df["department"].value_counts()
    return


@app.cell
def _(df, pl):
    # Since using Select, results in a DataFrame
    sorted_zip = (
                    df
                    .filter(pl.col('Postal Code').str.len_chars()<5)
                    .select('Postal Code')
                    .unique()
                    .sort('Postal Code')
    )
    sorted_zip
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3. Selecting & Filtering

    Column selection: `select()`, `pl.col()`, `pl.all()`, `pl.exclude()`

    Row filtering: `filter()`, boolean expressions

    Slicing: row/column indexing with `[]`

    Getting values: `item()`, `row()`
    """
    )
    return


@app.cell
def _(sample_df):
    _single_col_df = sample_df.select("name") #.select() returns a dataframe, not a series
    print(type(_single_col_df))
    _single_col_df
    return


@app.cell
def _(sample_df):
    # Accessing a single column directly by name  (returns a Series object)
    _single_col_series = sample_df["name"]
    _single_col_series
    return


@app.cell
def _(sample_df):
    # Select multiple columns:
    _multi_cols = sample_df.select(["name", "age", "salary"])
    _multi_cols
    return


@app.cell
def _(pl, sample_df):
    # Row filtering
    print("Filter rows where age > 30:")
    _filtered_df = sample_df.filter(pl.col("age") > 30)
    _filtered_df
    return


@app.cell
def _(pl, sample_df):
    print("\nFilter rows with multiple conditions: Engineers over 30")
    _complex_filter = sample_df.filter(
        (pl.col("age") > 30) & (pl.col("department") == "Engineering")
    )
    _complex_filter
    return


@app.cell
def _(sample_df):
    sample_df.write_csv('sample_df_employees.csv')
    return


@app.cell
def _(mo):
    mo.md(r"""# Sorting and Grouping""")
    return


@app.cell
def _(sample_df):
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
    return


@app.cell
def _(mo):
    mo.md(r"""# Grouping and aggregation""")
    return


@app.cell
def _(pl, sample_df):
    print("Group by department and calculate mean salary:")
    _grouped = sample_df.group_by("department").agg(
        pl.col("salary").mean().alias("avg_salary"),
        pl.col("age").mean().alias("avg_age"),
        pl.len().alias("count")
    )
    print(_grouped)
    return


@app.cell
def _(mo):
    mo.md(r"""Adding and Modifying Columns""")
    return


@app.cell
def _(pl, sample_df):
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
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Data Transformation

    Adding/modifying columns: `with_columns()`

    Expressions: `pl.col()` operations (math, string, date methods)
    sum_horizontal()
    max_horizontal()
    min_horizontal()
    mean_horizontal()

    Casting types: `cast()`

    Renaming: `alias()`, `rename()`

    Dropping: `drop()`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. String & Date Operations

    String methods: `str.to_lowercase()`, `str.contains()`, `str.replace()`

    Date parsing: `str.strptime()`, `str.to_date()`

    Date manipulation: `dt.year()`, `dt.month()`, date arithmetic, strptime(), strftime()
    """
    )
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
    print(parse_failures.select(["Order Date", "Order Date Parsed"]))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. Aggregations & Grouping

    Basic aggregations: `sum()`, `mean()`, `min()`, `max()`, `count()`

    Grouping: `group_by()` + `agg()`

    Multiple aggregations at once

    Window functions
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Sorting & Ranking

    Sorting: `sort()`, multiple columns, ascending/descending

    Ranking: `rank()`, `row_number()`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8. Joins & Concatenation

    Join types: `join()` (inner, left, outer, cross, semi, anti)

    Concatenating: `pl.concat()` (vertical/horizontal)

    Union operations
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9. Reshaping Data

    Pivoting: `pivot()`

    Melting: `melt()` (wide to long)

    Exploding lists: `explode()`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10. Conditional Logic

    `when().then().otherwise()` chains

    Complex conditional transformations

    `pl.lit()` for literal values
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 11. Missing Data

    Detecting: `is_null()`, `is_not_null()`, `null_count()`

    Handling: `fill_null()`, `drop_nulls()`, `forward_fill()`, `backward_fill()`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 12. Advanced Topics

    Lazy evaluation: `pl.LazyFrame`, `collect()`, query optimization

    Custom functions: `map_elements()` (use sparingly!)

    List columns: nested data structures

    Performance optimization techniques
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 13. I/O & Export

    Writing: `write_csv()`, `write_parquet()`, `write_excel()`

    Interoperability: converting to/from pandas, numpy, arrow
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
