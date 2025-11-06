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
    return go, mo, np, pl, px


@app.cell
def _(pl):
    df = pl.read_parquet('data/chicago_crime_2001_2025.parquet')
    df.shape
    return (df,)


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(mo):
    mo.md(r"""## Nice to have a data dictionary""")
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
    mo.md(r"""## How would we extract the quarter from the date into a new field called 'quater_period'?""")
    return


@app.cell
def _(df, pl):
    df2 = df.with_columns(pl.col('Date').str.to_datetime("%m/%d/%Y %I:%M:%S %p"))
    df2 = df2.with_columns(
        pl.col('Date').dt.year().alias('year'),
        pl.col('Date').dt.month().alias('month'),
        pl.col('Date').dt.day().alias('day'),
        pl.col('Date').dt.hour().alias('hour'),
        pl.col('Date').dt.minute().alias('minute'),
        pl.col('Date').dt.truncate('1mo').cast(pl.Date).alias('month_period')    
    )
    df2.glimpse()
    return (df2,)


@app.cell
def _(df2, pl):
    crimes_year_month = (df2
                        .group_by('month_period')
                        .agg(pl.col('ID')
                        .n_unique()
                        .alias('unique_ids')).sort('month_period')
                        )
    return (crimes_year_month,)


@app.cell
def _(crimes_year_month, px):
    fig_year_month = px.bar(crimes_year_month,
                            x='month_period',
                            y='unique_ids',
                            title='Crimes by Month',
                            labels={'ID':'Number of Unique Crimes', 'Month Period':'month_period'},
                            # markers=True
                            )
    fig_year_month.show()
    return


@app.cell
def _(df2):
    df2['hour'].value_counts()
    return


@app.cell
def _(mo):
    mo.md(r"""## Standardize the months""")
    return


@app.cell
def _(df2, pl):
    # Group month regardless of year, get count of crimes for each month
    unique_observations_by_month = (df2.group_by('month')
                                    .agg(pl.col('ID')
                                    .n_unique()
                                    .alias('unique_ids'))
                                    .sort('month')
    )
    unique_observations_by_month
    return (unique_observations_by_month,)


@app.cell
def _(px, unique_observations_by_month):
    # Which months typically have more crime?
    fig = px.bar(unique_observations_by_month, 
                  x='month', 
                  y='unique_ids',
                  title='Unique Crimes by Month',
                  labels={'ID': 'Number of Unique Crimes', 'Month': 'Month'},
                  text='unique_ids'
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
            pl.col('month').replace_strict(_days_in_month, default=None).cast(pl.Int64).alias('_days_in_month'),
            (pl.col('unique_ids') / pl.col('month').replace_strict(_days_in_month,default=None).cast(pl.Int32)).alias('normalized_crimes')))

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
    mo.md(r"""## Overlay temp on crimes per month""")
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
def _(
    chicago_monthly_avg_temp_with_windchill,
    pl,
    unique_observations_by_month_temp,
):
    # Chicago monthly average temperature with wind chill (°F)
    _chicago_monthly_avg_temp_with_windchill = {
        1: 23.3, 2: 27.0, 3: 37.7, 4: 48.0, 5: 58.4, 6: 68.4,
        7: 73.7, 8: 72.5, 9: 66.2, 10: 53.6, 11: 39.9, 12: 29.6
    }
    _temp = (unique_observations_by_month_temp
        .with_columns(pl.col('month')
        .replace_strict(chicago_monthly_avg_temp_with_windchill, default=None).alias('temperature'))
            )

    correlation = _temp.select(
        pl.corr('temperature','normalized_crimes').alias('correlation')).item()


    print(f'Correlation temp and crime: {correlation:.4f}')
    return


@app.cell
def _(df2):
    df2.columns
    return


@app.cell
def _(df2, np, pl):
    start_year = 2020
    end_year = 2024
    crimes_by_year = 100

    district_yearly_crimes = (
                             (df2.filter((pl.col('Year')>=start_year) & (pl.col('Year') <= end_year))
                            .group_by(['District', 'Year'])
                            .agg(pl.col('ID').n_unique().alias('crime_count'))
                            .sort(['District','Year'])
                             )
    )

    districts = district_yearly_crimes['District'].unique().sort()

    trend_results = []

    for district in districts:
        district_data = district_yearly_crimes.filter(pl.col('District') == district)

        years = district_data['Year'].to_numpy()
        crimes = district_data['crime_count'].to_numpy()

        if len(years) > 1:
            coefficients = np.polyfit(years, crimes, deg=1)
            slope = coefficients[0] # 0 = slope, 1 = y-intercept

            if slope > crimes_by_year:
                trend = 'Increasing'
            elif slope < -crimes_by_year:
                trend = 'Decreasing'
            else:
                trend = 'Stable'

        trend_results.append({
            'District': district,
            'Slope': slope,
            'Trend': trend,
            'Crime_change_per_year':slope
        })

        trends_df = pl.DataFrame(trend_results).sort('Slope',descending=True)

    print(f"Increasing: {trends_df.filter(pl.col('Trend') == 'Increasing').height} districts")
    print(f"Decreasing: {trends_df.filter(pl.col('Trend') == 'Decreasing').height} districts")
    print(f"Stable: {trends_df.filter(pl.col('Trend') == 'Stable').height} districts")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
