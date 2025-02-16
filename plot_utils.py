"""
Title: Plot Utilities  
Description: Provides styling and export functionalities for plots across the application. 
"""

# Standard Library Imports
import logging

# Third-Party Library Imports
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def remove_borders(plot_func):
    """
    A decorator to remove the borders around the plot area.
    It logs an informational message when the borders are removed.
    """

    def wrapper(*args, **kwargs):
        # Call the original plotting function
        fig, ax = plot_func(*args, **kwargs)  # Get both the figure and axes

        # Ensure we have an Axes object
        if not isinstance(ax, plt.Axes):
            raise ValueError("The plotting function must return a Matplotlib Axes object.")

        # Remove borders (spines) from the plot
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Log the border removal
        logger.info("Borders have been removed from the plot.")

        return fig, ax  # Return both the figure and axes object

    return wrapper

def standardize_time_series_ticks(plot_func):
    """
    A decorator to standardize the time series tick labels.
    It sets the tick frequency every 15 days, formats the labels, and rotates them at 90 degrees.
    Logs a warning if the x-axis is not of datetime type.
    """
    def wrapper(*args, **kwargs):
        # Call the original plotting function
        fig, ax = plot_func(*args, **kwargs)  # Get both the figure and axes

        # Ensure we have an Axes object
        if not isinstance(ax, plt.Axes):
            raise ValueError("The plotting function must return a Matplotlib Axes object.")

        # Check if the x-axis is of datetime type based on the data passed
        x_data = ax.get_lines()[0].get_xdata()
        if not isinstance(x_data[0], pd.Timestamp):
            logger.warning("X-axis is not of datetime type. Skipping tick adjustment.")
        else:
            # Set tick frequency to every 15 days
            tick_dates = pd.date_range(start=x_data[0], periods=15, freq='15D')
            ax.set_xticks(tick_dates)

            # Format ticks as datetime in a readable format (e.g., YYYY-MM-DD)
            ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in tick_dates], rotation=90)

        # Return the figure and axes objects
        return fig, ax

    return wrapper


def export_plot_to_disk(file_path: str):
    """
    A decorator to save the plot to disk as a PNG file.
    It always overwrites the file if it exists.

    Parameters:
        file_path (str): The path where the plot will be saved.
    """
    def decorator(plot_func):
        def wrapper(*args, **kwargs):
            # Call the original plotting function to get the figure and axis
            fig, ax = plot_func(*args, **kwargs)

            # Save the plot to the specified file path, always overwriting
            fig.savefig(file_path, format='png', bbox_inches='tight')
            print(f"Plot saved to {file_path}")

            # Return the figure and axis
            return fig, ax

        return wrapper
    return decorator


def apply_plot_styling(fig):
    """
    Apply consistent styling to a Plotly figure.
    """
    fig.update_layout(
        margin=dict(t=30, l=30, r=30, b=30),  # Margins
        legend=dict(x=0.75, y=1, bgcolor='rgba(255, 255, 255, 0.5)')  # Legend
    )
    return fig

