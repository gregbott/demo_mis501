import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import seaborn as sns
    import plotly.express as px
    sns.set_style('darkgrid') #'darkgrid' or 'whitegrid'
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import numpy as np
    return go, mo, np, pl, plt, px, sns


@app.cell
def _(pl):
    df = pl.read_parquet('data/chicago_crime_2001_2025.parquet')
    df.shape
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Review the data
    Learn what each field means. See the table below, which was copied from the https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data site.
    """
    )
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Dictionary""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Here's the data converted to a markdown table:

    | Column Name | Description | API Field Name | Data Type |
    |:-------------|:-------------|:----------------|:-----------|
    | ID | Unique identifier for the record. | id | Number |
    | Case Number | The Chicago Police Department RD Number (Records Division Number), which is unique to the incident. | case_number | Text |
    | Date | Date when the incident occurred. this is sometimes a best estimate. | date | Floating Timestamp |
    | Block | The partially redacted address where the incident occurred, placing it on the same block as the actual address. | block | Text |
    | IUCR | The Illinois Uniform Crime Reporting code. This is directly linked to the Primary Type and Description. See the list of IUCR codes at https://data.cityofchicago.org/d/c7ck-438e. | iucr | Text |
    | Primary Type | The primary description of the IUCR code. | primary_type | Text |
    | Description | The secondary description of the IUCR code, a subcategory of the primary description. | description | Text |
    | Location Description | Description of the location where the incident occurred. | location_description | Text |
    | Arrest | Indicates whether an arrest was made. | arrest | Checkbox |
    | Domestic | Indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act. | domestic | Checkbox |
    | Beat | Indicates the beat where the incident occurred. A beat is the smallest police geographic area – each beat has a dedicated police beat car. Three to five beats make up a police sector, and three sectors make up a police district. The Chicago Police Department has 22 police districts. See the beats at https://data.cityofchicago.org/d/aerh-rz74. | beat | Text |
    | District | Indicates the police district where the incident occurred. See the districts at https://data.cityofchicago.org/d/fthy-xz3r. | district | Text |
    | Ward | The ward (City Council district) where the incident occurred. See the wards at https://data.cityofchicago.org/d/sp34-6z76. | ward | Number |
    | Community Area | Indicates the community area where the incident occurred. Chicago has 77 community areas. See the community areas at https://data.cityofchicago.org/d/cauq-8yn6. | community_area | Text |
    | FBI Code | Indicates the crime classification as outlined in the FBI's National Incident-Based Reporting System (NIBRS). See the Chicago Police Department listing of these classifications at https://gis.chicagopolice.org/pages/crime_details. | fbi_code | Text |
    | X Coordinate | The x coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. This location is shifted from the actual location for partial redaction but falls on the same block. | x_coordinate | Number |
    | Y Coordinate | The y coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. This location is shifted from the actual location for partial redaction but falls on the same block. | y_coordinate | Number |
    | Year | Year the incident occurred. | year | Number |
    | Updated On | Date and time the record was last updated. | updated_on | Floating Timestamp |
    | Latitude | The latitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block. | latitude | Number |
    | Longitude | The longitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block. | longitude | Number |
    | Location | The location where the incident occurred in a format that allows for creation of maps and other geographic operations on this data portal. This location is shifted from the actual location for partial redaction but falls on the same block. | location | Location |
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Prep the data
    * Which fields, if any, need to be casted?
    * Given our questions, what additional fields do we need?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Date formats

    * %m - Month as number (10)

    * %d - Day of month (13)

    * %Y - 4-digit year (2025)

    * %I - Hour in 12-hour format (12)

    * %M - Minutes (00)

    * %S - Seconds (00)

    * %p - AM/PM indicator
    """
    )
    return


@app.cell
def _(df, pl):
    _df = df.with_columns(pl.col('Date').str.to_datetime("%m/%d/%Y %I:%M:%S %p"))
    _df.glimpse()
    return


@app.cell
def _():
    # df.write_parquet('chicago_crime_2001_2025.parquet')
    return


@app.cell
def _(mo):
    mo.md(r"""## Convert and Add New Columns for Analysis""")
    return


@app.cell
def _(df, pl, plt, sns):
    df2 = df.with_columns(pl.col('Date').str.to_datetime('%m/%d/%Y %I:%M:%S %p')) # Note 'datetime', NOT 'date'
    print(df2.schema)
    df2 = df2.with_columns(
        pl.col('Date').dt.year().alias('year'),
        pl.col('Date').dt.month().alias('month'),
        pl.col('Date').dt.truncate('1mo').cast(pl.Date).alias('month_period'),
        pl.col('Date').dt.truncate('1q').cast(pl.Date).alias('quarter_period'),
        pl.col('Date').dt.day().alias('day'),
        pl.col('Date').dt.hour().alias('hour'),
        pl.col('Date').dt.minute().alias('minute'),
    )
    df2.glimpse()
    print(df2['hour'].value_counts().sort('hour'))

    # Count occurrences by hour
    hour_counts = df2.group_by("hour").len().sort("hour")

    # Create bar plot
    sns.barplot(data=hour_counts, x="hour", y="len")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Crimes")
    plt.title("Distribution of Crimes by Hour")
    plt.show()
    return (df2,)


@app.cell
def _(df2):
    df2.glimpse()
    return


@app.cell
def _(df2, plt, sns):
    # Count occurrences by hour
    minute_counts = df2.group_by("minute").len().sort("minute")

    # Create bar plot
    plt.figure(figsize=(14,6))
    sns.barplot(data=minute_counts, x="minute", y="len")
    plt.xlabel("Minute of Hour")
    plt.ylabel("Number of Crimes")
    plt.title("Distribution of Crimes by Minute of Hour")
    plt.show()
    return


@app.cell
def _(df2, pl):
    unique_observations_by_month = (df2.group_by('month')
                                    .agg(pl.col('ID')
                                    .n_unique()
                                    .alias('unique_ids'))
                                    .sort('month')
                                   )
    return (unique_observations_by_month,)


@app.cell
def _(mo):
    mo.md(r"""## Which months are higher or lower for crime in Chicago?""")
    return


@app.cell
def _(px, unique_observations_by_month):
    # Which months typically have more crime?
    fig = px.bar(unique_observations_by_month, 
                  x='month', 
                  y='unique_ids',
                  title='Unique Crimes by Month',
                  labels={'ID': 'Number of Unique Crimes', 'Month': 'Month'},
                  text='unique_ids'  # Add this line
                )
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')  # Changed to .0f for whole numbers
    fig.show()
    return


@app.cell
def _(df2, pl, px):
    # Define days in each month (non-leap year)
    _days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    _unique_observations_by_month_std = (
        df2.group_by('month')
        .agg(pl.col('ID').n_unique().alias('unique_ids'))
        .sort('month')
        .with_columns(
            pl.col('month').replace_strict(_days_in_month, default=None).cast(pl.Int64).alias('days_in_month'),
            (pl.col('unique_ids') / pl.col('month').replace_strict(_days_in_month, default=None).cast(pl.Int32)).alias('normalized_crimes')
        )
    )
    _fig_std = px.bar(_unique_observations_by_month_std, 
                  x='month', 
                  y='normalized_crimes',
                  title='Unique Crimes by Month',
                  labels={'ID': 'Number of Unique Crimes', 'Month': 'Month'},
                  text='normalized_crimes'  # Add this line to display values
                )
    _fig_std.update_traces(texttemplate='%{text:.0f}', textposition='outside')  # Format and position the text
    _fig_std.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## How does temperature correlate with months?""")
    return


@app.cell
def _(df2, go, pl):
    # Define days in each month (non-leap year)
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }

    # Chicago monthly average temperature with wind chill (°F)
    chicago_monthly_avg_temp_with_windchill = {
        1: 23.3, 2: 27.0, 3: 37.7, 4: 48.0, 5: 58.4, 6: 68.4,
        7: 73.7, 8: 72.5, 9: 66.2, 10: 53.6, 11: 39.9, 12: 29.6
    }

    # Process crime data
    unique_observations_by_month_temp = (
        df2.group_by('month')
        .agg(pl.col('ID').n_unique().alias('unique_ids'))
        .sort('month')
        .with_columns(
            pl.col('month').replace_strict(days_in_month, default=None).cast(pl.Int32).alias('days_in_month'),
            (pl.col('unique_ids') / pl.col('month').replace_strict(days_in_month, default=None).cast(pl.Int32)).alias('normalized_crimes')
        )
    )

    # Create chart with bar chart for crimes and line for temperature
    fig_temp = go.Figure()

    # Add normalized crimes bar chart
    fig_temp.add_trace(
        go.Bar(
            x=unique_observations_by_month_temp['month'],
            y=unique_observations_by_month_temp['normalized_crimes'],
            name='Normalized Crimes per Day',
            marker=dict(color='blue'),
            yaxis='y1'
        )
    )

    # Add temperature line
    fig_temp.add_trace(
        go.Scatter(
            x=list(chicago_monthly_avg_temp_with_windchill.keys()),
            y=list(chicago_monthly_avg_temp_with_windchill.values()),
            name='Avg Temp with Wind Chill (°F)',
            line=dict(color='red'),
            yaxis='y2'
        )
    )

    # Update layout for dual y-axes
    fig_temp.update_layout(
        title='Chicago Normalized Crime Rate (Bars) vs Average Temperature with Wind Chill (Line) by Month',
        xaxis=dict(title='Month', 
                   tickmode='array', 
                   tickvals=list(range(1, 13))),
        yaxis=dict(title='Normalized Crimes per Day', 
                   title_font=dict(color='blue'), 
                   tickfont=dict(color='blue')),
        yaxis2=dict(title='Temperature (°F)', 
                    title_font=dict(color='red'), 
                    tickfont=dict(color='red'), 
                    overlaying='y', 
                    side='right', 
                    range=[0, 100]),
        legend=dict(x=0.1, y=1.1, orientation='h')
    )

    # Show the plot
    fig_temp.show()
    return (
        chicago_monthly_avg_temp_with_windchill,
        unique_observations_by_month_temp,
    )


@app.cell
def _():
    ## Check actual correlation
    return


@app.cell
def _(
    chicago_monthly_avg_temp_with_windchill,
    pl,
    unique_observations_by_month_temp,
):
    # Add temperature data to the dataframe
    _unique_observations_by_month_temp = unique_observations_by_month_temp.with_columns(
        pl.col('month').replace_strict(chicago_monthly_avg_temp_with_windchill, default=None).alias('temperature')
    )

    # Calculate correlation between temperature and normalized crimes
    correlation = _unique_observations_by_month_temp.select(
        pl.corr('temperature', 'normalized_crimes').alias('correlation')
    ).item()

    print(f"Correlation between temperature and normalized crimes: {correlation:.4f}")
    return


@app.cell
def _(df2, pl):
    crimes_year_month = (df2
                        .group_by('month_period')
                        .agg(pl.col('ID')
                            .n_unique()
                            .alias('unique_ids'))
                        .sort('month_period')
                        )
    return (crimes_year_month,)


@app.cell
def _(crimes_year_month, px):
    fig_year_month = px.line(crimes_year_month, 
                  x='month_period', 
                  y='unique_ids',
                  title='Crimes by Year, Month',
                  labels={'ID': 'Number of Unique Crimes', 'Month Period': 'month_period'},
                  markers=True
                )  # Add markers to points

    fig_year_month.show()
    return


@app.cell
def _(df2, pl):
    crime_by_type = (
        df2
        .filter(pl.col('year') == 2025)
        .group_by('Primary Type')
        .agg(
            pl.len().alias('total_crimes')
        )
        .sort('total_crimes', descending=True)
    )

    crime_by_type
    return


@app.cell
def _(df2, pl, px):
    # Filter by specific crime types and sum by year
    crime_types = ['HOMICIDE', 'CRIMINAL SEXUAL ASSAULT', 'PROSTITUTION']

    filtered_crimes = (
        df2
        .filter(pl.col('Primary Type').is_in(crime_types))
        .group_by(['year', 'Primary Type'])
        .agg(pl.len().alias('total_crimes'))
        .sort('year')
    )

    # Create line chart with separate line for each crime type
    serious_crime_fig = px.line(filtered_crimes, 
                  x='year', 
                  y='total_crimes',
                  color='Primary Type',
                  title='Serious Crimes by Year',
                  labels={'total_crimes': 'Number of Crimes', 
                          'year': 'Year',
                          'Primary Type': 'Crime Type'},
                  markers=True)

    serious_crime_fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        legend_title="Crime Type"
    )

    serious_crime_fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## How has the total number of cases evolved over time?""")
    return


@app.cell
def _(df2):
    df2.columns
    return


@app.cell
def _(df2, pl, px):
    # Group by date and count cases
    cases_by_month = (
        df2
        .group_by("month_period")
        .agg(pl.count().alias("Total_Cases"))
        .sort("month_period")
    )

    # View the results
    print(cases_by_month)

    # Optional: Add cumulative sum to see total cases accumulated over time
    # cases_over_time = cases_over_time.with_columns(
    #     pl.col("Total_Cases").cum_sum().alias("Cumulative_Cases")
    # )

    print(cases_by_month)

    _fig = px.line(cases_by_month, 
                  x='month_period', 
                  y='Total_Cases',
                  title='Total Cases Over Time')
    _fig.show()
    return


@app.cell
def _(df2, pl, px):
    # Group by date and count cases
    cases_by_quarter = (
        df2
        .group_by("quarter_period")
        .agg(pl.count().alias("Total_Cases"))
        .sort("quarter_period")
    )

    # View the results
    print(cases_by_quarter)

    _fig = px.line(cases_by_quarter, 
                  x='quarter_period', 
                  y='Total_Cases',
                  title='Total Cases By Quarter')
    _fig.show()
    return


@app.cell
def _(df2, pl, px):
    # Group by date and count cases
    cases_by_year = (
        df2
        .group_by("year")
        .agg(pl.count().alias("Total_Cases"))
        .sort("year")
    )

    # View the results
    print(cases_by_year)

    _fig = px.line(cases_by_year, 
                  x='year', 
                  y='Total_Cases',
                  title='Total Cases By Year')
    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How has the proportion of cases leading to an arrest changed over time?
    How has the Primary Type of call evolved over time?
    Are there any observable patterns in arrest rates by district?
    Have any specific types of crime (```IUCR```s) changed dramatically over time?
    How is crime affected by the hour of the day?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## How have ```Domestic``` crimes evolved over time?""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### First, understand the 'domestic data'

    * Re-read the data dictionary
    * Based on the data dictionary, how should you learn more about what 'Domestic means'?
    """
    )
    return


@app.cell
def _(df2, pl, px):
    domestic_cases = (
                        df2
                        .filter(pl.col('Domestic')==True)
                        .group_by('Primary Type')
                        .agg(pl.len().alias('count'))
                        .sort('count',descending=True)
                        .head(10)
    )
    domestic_cases

    _fig = px.pie(
        domestic_cases,
        values='count',
        names='Primary Type',
        title='Top 10 Domestic Crime Types'
    )

    # _fig.update_traces(
    #     textposition='inside',
    #     textinfo='percent+label',
    #     pull=[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Optional: pull out the largest slice
    # )


    _fig.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## District Crime Trends""")
    return


@app.cell
def _(df2, np, pl):
    start_year = 2020
    end_year = 2024
    crimes_per_year = 100 # What does 'increasing' or 'decreasing' mean? This number sets the threshold of crimes

    # Filter data for years 2020-2024 and group by district and year
    district_yearly_crimes = (
        df2.filter((pl.col('Year') >= start_year) & (pl.col('Year') <= end_year))
        .group_by(['District', 'Year'])
        .agg(pl.col('ID').n_unique().alias('crime_count'))
        .sort(['District', 'Year'])
    )

    # Get unique districts
    districts = district_yearly_crimes['District'].unique().sort()

    # Store results
    trend_results = [] # Create a list to contain the District, Slope, Trend, and change per year (slope)

    # Analyze each district
    for district in districts:
        # Filter data for this district
        district_data = district_yearly_crimes.filter(pl.col('District') == district)

        # Extract years and crime counts
        years = district_data['Year'].to_numpy()
        crimes = district_data['crime_count'].to_numpy()

        # Fit linear trend (degree=1)
        if len(years) > 1:  # Need at least 2 points for a line
            coefficients = np.polyfit(years, crimes, deg=1) # deg 1 means linear
            slope = coefficients[0] # coefficient 0 is the slope (m), 1 is the y-intercept (b)

            # Classify trend
            if slope > crimes_per_year:  # Increasing by more than 100 crimes/year
                trend = 'Increasing'
            elif slope < -crimes_per_year:  # Decreasing by more than 100 crimes/year
                trend = 'Decreasing'
            else:
                trend = 'Stable'

            trend_results.append({
                'District': district,
                'Slope': slope,
                'Trend': trend,
                'Crime_Change_Per_Year': slope
            })

    # Convert to Polars DataFrame
    trends_df = pl.DataFrame(trend_results).sort('Slope', descending=True)

    print("\n=== Crime Trends by District (2020-2024) ===\n")
    print(trends_df)

    # Summary statistics
    print("\n=== Summary ===")
    # height = num of rows, width = number of columns, shape is both
    print(f"Increasing: {trends_df.filter(pl.col('Trend') == 'Increasing').height} districts") 
    print(f"Decreasing: {trends_df.filter(pl.col('Trend') == 'Decreasing').height} districts")
    print(f"Stable: {trends_df.filter(pl.col('Trend') == 'Stable').height} districts")

    # Show top 5 increasing and decreasing districts
    print("\n=== Top 5 Districts with Increasing Crime ===")
    print(trends_df.filter(pl.col('Trend') == 'Increasing').head(5))

    print("\n=== Top 5 Districts with Decreasing Crime ===")
    print(trends_df.filter(pl.col('Trend') == 'Decreasing').head(5))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
