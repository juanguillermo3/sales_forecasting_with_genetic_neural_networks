"""
Title: Results Parser  
Description: Generates visual reports from optimization process results.  
Depends on: plot_utils.py
image_path: results_*.png
"""

import pandas as pd
import matplotlib.pyplot as plt

from plot_utils import export_plot_to_disk

class OptimizationResultsParser:
    def __init__(self, file_path: str, rounding_precision: int = 3):
        """Initialize the parser, load the Excel file, and ingest the data."""
        self.file_path = file_path
        self.rounding_precision = rounding_precision  # Configurable rounding precision
        self.dfs = pd.read_excel(file_path, sheet_name=None)  # Load all sheets

    def _round_param(self, param_value):
        """Helper method to round a numeric value to the desired precision."""
        try:
            return round(float(param_value), self.rounding_precision)
        except (ValueError, TypeError):
            return param_value  # Return as-is if not a valid numeric value

    def _parse_optimization_parameters(self):
        """Extract and round optimization parameters."""
        df = self.dfs.get("Optimization Parameters")
        if df is not None:
            return {row["Parameter"]: self._round_param(row["Value"]) for _, row in df.iterrows()}
        return {}

    def _parse_best_solution(self):
        """Extract and round best solution's hyperparameters and fitness value."""
        df = self.dfs.get("Best Solution")
        if df is not None:
            return {row["Parameter"]: self._round_param(row["Value"]) for _, row in df.iterrows()}
        return {}

    @export_plot_to_disk("results_best_forecast.png")
    def plot_best_forecast(self, ax):
        """Plot historical vs. forecasted values as a time series."""
        df = self.dfs.get("Best Forecast")
        if df is not None:
            ax.plot(df.index, df["Historical"], label="Historical", linestyle='-', color='black')
            ax.plot(df.index, df["Forecasts"], label="Forecast", linestyle='--', color='red')
            ax.set_title("Best Forecast vs. Actual Values")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
        return ax.figure, ax


    @export_plot_to_disk("results_optimization.png")
    def plot_general_results(self, ax):
        """Plot the 25th, 50th, and 75th percentile trends over generations, including best solution."""
        df = self.dfs.get("General Results-r2")
        if df is not None:
            ax.plot(df["Generation"], df["25th Percentile"], label="25th Percentile", linestyle='dotted', color='blue')
            ax.plot(df["Generation"], df["50th Percentile"], label="50th Percentile (Median)", linestyle='solid', color='green')
            ax.plot(df["Generation"], df["75th Percentile"], label="75th Percentile", linestyle='dashed', color='orange')
            ax.plot(df["Generation"], df["Max Fitness"], label="Best Solution", linestyle='dashdot', color='purple')
            ax.set_title("Optimization Performance Over Generations")
            ax.set_xlabel("Generation")
            ax.set_ylabel("R² Fitness")
            ax.legend()
            ax.grid(True)
        return ax.figure, ax


    def _generate_caption(self, figure_number="X"):
        """Generate a structured caption for the report."""
        params = self._parse_optimization_parameters()
        best_solution = self._parse_best_solution()

        # Extract key values
        pop_size = params.get("Population Size", "N/A")
        generations = params.get("Number of Generations", "N/A")
        mutation_rate = params.get("Mutation Rate", "N/A")
        crossover_rate = params.get("Crossover Rate", "N/A")
        fitness = best_solution.get("Fitness", "N/A")

        # Construct statements
        general_statement = (f"A population of {pop_size} neural network architectures "
                             f"was evolved over {generations} generations with a mutation rate of {mutation_rate} "
                             f"and a crossover rate of {crossover_rate}.")
        best_solution_statement = (f"Best solution found: {', '.join([f'{k}: {v}' for k, v in best_solution.items() if k != 'Elapsed Time'])}. "
                                   f"Final fitness: {fitness}.")

        caption = f"{general_statement} {best_solution_statement}"

        return caption

    def render_report(self, output_file="optimization_report.pdf", figure_number="X"):
        """Generate and save a unified PDF report combining key optimization insights."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # Two vertical plots

        self.plot_best_forecast(axes[0])  # First plot
        self.plot_general_results(axes[1])  # Second plot

        plt.tight_layout()

        # Increase spacing and add the caption to the figure
        caption = self._generate_caption(figure_number)
        plt.figtext(0.1, 0.01, caption, ha='left', va='top', fontsize=9, color='gray',
                    fontname='serif', wrap=True, horizontalalignment='center', verticalalignment='top',
                    bbox={'facecolor': 'white', 'alpha': 0.0, 'edgecolor': 'none', 'pad': 15})

        plt.savefig(output_file, format='pdf')
        plt.show()
        plt.close()

        # Print caption as confirmation
        print(f"Report saved to {output_file}")
