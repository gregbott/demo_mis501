import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Chicago Homicides by District (2022-2024)

    This notebook analyzes homicide patterns across Chicago police districts
    for the years 2022, 2023, and 2024 using data from the Chicago Police Department.

    ## Overview

    We'll create a pivot table showing:
    - **Rows**: Police districts (1-25)
    - **Columns**: Years (2022, 2023, 2024)
    - **Values**: Count of homicides

    This analysis helps identify which districts have the highest homicide rates
    and how these rates are changing over time.
    """
    )
    return


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(mo):
    mo.md("""## Loading Chicago Crime Data""")
    return


@app.cell
def _(pl):
    # Load the Chicago crime data
    crime_df = pl.read_parquet('data/chicago_crime_2001_2025.parquet')

    # Display basic information
    total_records = len(crime_df)
    date_range = f"{crime_df['Year'].min()} to {crime_df['Year'].max()}"

    crime_df.head()
    return crime_df, date_range, total_records


@app.cell
def _(date_range, mo, total_records):
    mo.md(
        f"""
    **Dataset Information:**
    - Total crime records: {total_records:,}
    - Date range: {date_range}
    - Source: Chicago Police Department
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""## Filtering for Homicides (2022-2024)""")
    return


@app.cell
def _(crime_df, pl):
    # Filter for homicides in years 2022, 2023, and 2024
    homicides = crime_df.filter(
        (pl.col('Primary Type') == 'HOMICIDE') &
        (pl.col('Year').is_in([2022, 2023, 2024]))
    )

    # Get statistics
    total_homicides = len(homicides)
    unique_districts = homicides['District'].n_unique()

    homicides.head(10)
    return homicides, total_homicides, unique_districts


@app.cell
def _(mo, total_homicides, unique_districts):
    mo.md(
        f"""
    **Filtered Data:**
    - Total homicides (2022-2024): {total_homicides:,}
    - Number of districts: {unique_districts}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Pivot Table: Homicides by District and Year

    This pivot table shows the count of homicides for each district across the three years.
    """
    )
    return


@app.cell
def _(homicides):
    # Create pivot table with districts as rows and years as columns
    pivot_homicides = homicides.pivot(
        values='ID',  # Count occurrences
        index='District',
        on='Year',  # Updated from 'columns' to 'on'
        aggregate_function='len'  # Updated from 'count' to 'len'
    ).sort('District')

    # Fill null values with 0 (districts with no homicides in a given year)
    pivot_homicides_clean = pivot_homicides.fill_null(0)

    # Rename columns for clarity
    pivot_homicides_display = pivot_homicides_clean.rename({
        '2022': 'Homicides_2022',
        '2023': 'Homicides_2023',
        '2024': 'Homicides_2024'
    })
    pivot_homicides_display
    return (pivot_homicides_display,)


@app.cell
def _(pivot_homicides_display, pl):
    # Add a total column
    pivot_homicides_final = pivot_homicides_display.with_columns([
        (pl.col('Homicides_2022') + pl.col('Homicides_2023') + pl.col('Homicides_2024')).alias('Total_2022_2024')
    ])

    pivot_homicides_final
    return (pivot_homicides_final,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary Statistics

    Let's calculate some key metrics from our pivot table.
    """
    )
    return


@app.cell
def _(pivot_homicides_final, pl):
    # Calculate summary statistics
    total_by_year = {
        '2022': pivot_homicides_final['Homicides_2022'].sum(),
        '2023': pivot_homicides_final['Homicides_2023'].sum(),
        '2024': pivot_homicides_final['Homicides_2024'].sum()
    }

    # Find district with most homicides overall
    max_district_row = pivot_homicides_final.sort('Total_2022_2024', descending=True).head(1)
    max_district = max_district_row['District'].item()
    max_total = max_district_row['Total_2022_2024'].item()

    # Find district with least homicides overall (excluding zeros)
    min_district_row = pivot_homicides_final.filter(
        pl.col('Total_2022_2024') > 0
    ).sort('Total_2022_2024').head(1)
    min_district = min_district_row['District'].item()
    min_total = min_district_row['Total_2022_2024'].item()

    summary_stats = {
        'total_by_year': total_by_year,
        'max_district': max_district,
        'max_total': max_total,
        'min_district': min_district,
        'min_total': min_total
    }

    summary_stats
    return max_district, max_total, min_district, min_total, total_by_year


@app.cell
def _(max_district, max_total, min_district, min_total, mo, total_by_year):
    mo.md(
        f"""
    ### Key Findings

    **Total Homicides by Year:**
    - 2022: {total_by_year['2022']} homicides
    - 2023: {total_by_year['2023']} homicides
    - 2024: {total_by_year['2024']} homicides

    **District Analysis (2022-2024 combined):**
    - **Highest**: District {max_district} with {max_total} homicides
    - **Lowest**: District {min_district} with {min_total} homicides

    **Trend**: {'↓ Decreasing' if total_by_year['2024'] < total_by_year['2022'] else '↑ Increasing' if total_by_year['2024'] > total_by_year['2022'] else '→ Stable'}
    (2022: {total_by_year['2022']} → 2024: {total_by_year['2024']})
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Top 10 Districts by Total Homicides

    The districts with the highest number of homicides over the three-year period:
    """
    )
    return


@app.cell
def _(pivot_homicides_final):
    # Get top 10 districts
    top_10_districts = pivot_homicides_final.sort('Total_2022_2024', descending=True).head(10)
    top_10_districts
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Year-over-Year Changes

    Let's calculate the change in homicides from year to year for each district.
    """
    )
    return


@app.cell
def _(pivot_homicides_final, pl):
    # Calculate year-over-year changes
    yoy_changes = pivot_homicides_final.with_columns([
        (pl.col('Homicides_2023') - pl.col('Homicides_2022')).alias('Change_2022_to_2023'),
        (pl.col('Homicides_2024') - pl.col('Homicides_2023')).alias('Change_2023_to_2024'),
        (pl.col('Homicides_2024') - pl.col('Homicides_2022')).alias('Change_2022_to_2024')
    ])

    yoy_changes
    return (yoy_changes,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Districts with Largest Changes

    Let's identify districts with the most significant increases or decreases.
    """
    )
    return


@app.cell
def _(yoy_changes):
    # Districts with largest increase (2022 to 2024)
    largest_increase = yoy_changes.sort('Change_2022_to_2024', descending=True).head(5)

    # Districts with largest decrease (2022 to 2024)
    largest_decrease = yoy_changes.sort('Change_2022_to_2024').head(5)

    print("Districts with Largest Increase (2022 to 2024):")
    print(largest_increase)
    print("\nDistricts with Largest Decrease (2022 to 2024):")
    print(largest_decrease)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Conclusion

    This analysis provides insights into homicide patterns across Chicago police districts
    from 2022 to 2024. The pivot table allows us to quickly identify:

    1. **High-risk districts** that may need additional resources
    2. **Temporal trends** showing whether homicides are increasing or decreasing
    3. **Year-over-year changes** to track the effectiveness of interventions

    **Note**: This data is for educational and analytical purposes. For the most current
    crime statistics and safety information, please consult the Chicago Police Department
    directly.
    """
    )
    return


if __name__ == "__main__":
    app.run()
