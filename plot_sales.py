"""
Title: Plot Sales
Description: Plots the hourly sales for the food delivery service
"""

import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import remove_borders, standardize_time_series_ticks, export_plot_to_disk

@remove_borders
@standardize_time_series_ticks
@export_plot_to_disk("hourly_sales.png")
def plot_hourly_sales(df: pd.DataFrame):
    """
    Plots the hourly sales for a food delivery service.

    Parameters:
        df (pd.DataFrame): A DataFrame containing 'datetime' and 'sales' columns.

    Raises:
        ValueError: If required columns are missing or 'datetime' is not in datetime format.
    """
    # Rename 'value' to 'sales' if present
    df = df.rename(columns={'value': 'sales'})

    # Ensure required columns exist
    required_columns = {'datetime', 'sales'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Ensure 'datetime' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Set 'datetime' as index without dropping
    df_for_plot = df.set_index('datetime', drop=False)

    # Determine date range and frequency
    start_date = df_for_plot['datetime'].min().strftime('%B %d, %Y')  # Full month name, day, year
    end_date = df_for_plot['datetime'].max().strftime('%B %d, %Y')
    freq = pd.infer_freq(df_for_plot['datetime'])

    # Define a more user-friendly frequency string
    if freq == 'H':
        freq_str = 'Hourly'
    elif freq == 'D':
        freq_str = 'Daily'
    else:
        freq_str = 'Irregular'

    subtitle = f"From {start_date} to {end_date} | Frequency: {freq_str}"

    # Plot the data using ax for direct manipulation
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_for_plot.index, df_for_plot['sales'], label='Hourly Sales', color='tab:blue')

    # Customizing the plot
    ax.set_title('Hourly Sales Trend for Food Delivery Services', fontsize=14, pad=20)
    ax.set_xlabel('Date & Time')
    ax.set_ylabel('Sales Volume')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Add subtitle below title
    ax.text(0.5, 1.02, subtitle, ha='center', va='bottom', transform=ax.transAxes, fontsize=10, color='gray')

    # Show the plot
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    #plt.show()

    return fig, ax  # Return both the figure and Axes object
