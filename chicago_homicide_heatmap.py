import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import polars as pl
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import LinearColorMapper, ColorBar, HoverTool, BasicTicker
    from bokeh.palettes import YlOrRd9
    from bokeh.transform import linear_cmap
    from bokeh.io import output_notebook
    import numpy as np

    # Configure bokeh to work in notebooks
    output_notebook()

    return pl, figure, output_file, show, LinearColorMapper, ColorBar, HoverTool, BasicTicker, YlOrRd9, linear_cmap, output_notebook, np


@app.cell
def __(mo):
    return mo.md("""
    # 🔴 Chicago Homicide Heat Map

    This visualization shows the geographic distribution of homicides in Chicago using a yellow-orange-red gradient,
    where darker red indicates higher homicide frequency.
    """)


@app.cell
def __(pl):
    # Load the Chicago crime data
    parquet_path = "./data/chicago_crime_2001_2025.parquet"

    # Load and filter for homicides only
    homicide_df = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("Primary Type") == "HOMICIDE")
        .filter(
            (pl.col("Latitude").is_not_null()) &
            (pl.col("Longitude").is_not_null())
        )
        .select([
            "Latitude",
            "Longitude",
            "Date",
            "Year",
            "Description",
            "Location Description",
            "Block"
        ])
        .collect()
    )

    print(f"Total homicides with location data: {homicide_df.shape[0]:,}")

    return parquet_path, homicide_df


@app.cell
def __(pl, homicide_df):
    # Group by location to get frequency for heat map intensity
    location_freq = (
        homicide_df
        .group_by(["Latitude", "Longitude"])
        .agg([
            pl.len().alias("count"),
            pl.col("Description").first().alias("description"),
            pl.col("Location Description").first().alias("location_desc"),
            pl.col("Block").first().alias("block"),
            pl.col("Year").min().alias("first_year"),
            pl.col("Year").max().alias("last_year")
        ])
        .sort("count", descending=True)
    )

    # Convert to pandas for bokeh
    location_pd = location_freq.to_pandas()

    print(f"Unique homicide locations: {len(location_pd):,}")
    print(f"\nTop 10 locations by frequency:")
    print(location_freq.head(10))

    return location_freq, location_pd


@app.cell
def __(location_pd, figure, LinearColorMapper, ColorBar, HoverTool, BasicTicker, YlOrRd9, linear_cmap):
    # Create the heat map using Bokeh

    # Set up the output
    output_file("chicago_homicide_heatmap.html")

    # Create figure with appropriate range
    p = figure(
        width=1000,
        height=800,
        title="Chicago Homicide Heat Map (Yellow-Orange-Red Gradient)",
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
        x_axis_label="Longitude",
        y_axis_label="Latitude",
        toolbar_location="above"
    )

    # Create color mapper with yellow-orange-red gradient
    # YlOrRd9 = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
    mapper = LinearColorMapper(
        palette=YlOrRd9,
        low=location_pd['count'].min(),
        high=location_pd['count'].max()
    )

    # Add circle markers with color based on frequency
    circles = p.circle(
        x='Longitude',
        y='Latitude',
        size='size',
        source=location_pd.assign(
            size=np.clip(location_pd['count'] * 3 + 5, 5, 30)  # Scale size based on frequency
        ),
        fill_color={'field': 'count', 'transform': mapper},
        fill_alpha=0.7,
        line_color="darkred",
        line_width=0.5,
        line_alpha=0.5
    )

    # Configure hover tool
    hover = p.select_one(HoverTool)
    hover.tooltips = [
        ("Location", "@block"),
        ("Homicides", "@count"),
        ("Type", "@location_desc"),
        ("Years", "@first_year - @last_year"),
        ("Lat, Lon", "(@Latitude{0.0000}, @Longitude{0.0000})")
    ]

    # Add color bar
    color_bar = ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(desired_num_ticks=9),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title="Homicide\nFrequency"
    )
    p.add_layout(color_bar, 'right')

    # Style the plot
    p.background_fill_color = "#f5f5f5"
    p.grid.grid_line_color = "white"
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None

    # Display the plot
    show(p)

    return p, mapper, color_bar, circles, hover


@app.cell
def __(mo, homicide_df, location_freq):
    # Statistics summary
    total_homicides = homicide_df.shape[0]
    unique_locations = location_freq.shape[0]
    max_freq_location = location_freq.row(0, named=True)
    min_year = homicide_df.select(pl.col("Year").min()).item()
    max_year = homicide_df.select(pl.col("Year").max()).item()

    return mo.md(f"""
    ## 📊 Homicide Statistics

    - **Total Homicides**: {total_homicides:,}
    - **Unique Locations**: {unique_locations:,}
    - **Date Range**: {min_year} - {max_year}
    - **Highest Frequency Location**: {max_freq_location['block']} ({max_freq_location['count']} homicides)

    ## 🗺️ Heat Map Features

    - **Color Scale**: Yellow (low frequency) → Orange (medium) → Red (high frequency)
    - **Point Size**: Larger circles indicate more homicides at that location
    - **Interactive**: Hover over points to see detailed information
    - **Tools**: Pan, zoom, and reset to explore different areas

    ## 💡 Interpretation

    - **Yellow areas**: 1-2 homicides at the location
    - **Orange areas**: 3-5 homicides at the location
    - **Red areas**: 6+ homicides at the location (highest risk areas)
    - Darker red and larger circles indicate the most dangerous locations
    """)


@app.cell
def __(pl, homicide_df):
    # Year-over-year analysis
    homicides_by_year = (
        homicide_df
        .group_by("Year")
        .agg(pl.len().alias("Homicides"))
        .sort("Year")
    )

    homicides_by_year


@app.cell
def __(mo):
    return mo.md("""
    ---

    ### 📁 Output File

    The interactive heat map has been saved to: **chicago_homicide_heatmap.html**

    You can open this file in any web browser to explore the interactive visualization.
    """)


if __name__ == "__main__":
    app.run()
