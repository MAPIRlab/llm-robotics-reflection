import matplotlib.pyplot as plt
import numpy as np

import constants
from constants import (
    METHOD_BASE,
    METHOD_ENSEMBLING,
    METHOD_MULTIAGENT_REFLECTION,
    METHOD_SELF_REFLECTION,
)


class ChartSemanticMapComplexityGenerator:
    """
    Generates a chart showing the relationship between semantic map complexity and a specified metric.
    """

    def __init__(
        self,
        all_comparison_results_df,
        semantic_map_basenames,
        semantic_map_sizes,
        mode,
        llm_provider_name
    ):
        """
        Initializes the chart generator with the necessary data.

        Parameters:
        - all_comparison_results_df (DataFrame): DataFrame containing all comparison results.
        - semantic_map_basenames (list): List of semantic map base names.
        - semantic_map_sizes (list): List of corresponding semantic map sizes.
        - mode (str): The mode to filter the data.
        - llm_provider_name (str): The name of the LLM provider to filter the data.
        """
        self.df = all_comparison_results_df.copy()
        self.semantic_map_size_dict = dict(
            zip(semantic_map_basenames, semantic_map_sizes))
        self.mode = mode
        self.llm_provider_name = llm_provider_name

    def generate_chart(self, metric):
        """
        Generates and displays a chart based on the specified metric.

        Parameters:
        - metric (str): The performance metric to plot ('top_1', 'top_2', 'top_3', or 'top_any').
        """
        # Step 1: Filter DataFrame by LLM provider and mode
        df_filtered = self._filter_dataframe()

        # Step 2: Map semantic map basenames to sizes and add as a new column
        df_filtered = self._add_semantic_map_size(df_filtered)

        # Step 3: Aggregate comparison results by method and semantic map
        df_grouped = self._aggregate_comparison_results(df_filtered)

        # Step 4: Compute the desired metric rates
        df_grouped = self._compute_metric_rates(df_grouped)

        # Step 5: Prepare data for plotting
        df_grouped = self._prepare_plotting_data(df_grouped, metric)

        # Step 6: Plot the chart
        self._plot_chart(df_grouped, metric)

    def _filter_dataframe(self):
        """
        Filters the DataFrame to include only rows matching the specified LLM provider and mode.
        """
        return self.df[
            (self.df['LLM'] == self.llm_provider_name) &
            (self.df['Mode'] == self.mode)
        ]

    def _add_semantic_map_size(self, df):
        """
        Adds a 'SemanticMapSize' column to the DataFrame by mapping semantic map names to their sizes.
        """
        df['SemanticMapSize'] = df['SemanticMap'].map(
            self.semantic_map_size_dict)
        return df

    def _aggregate_comparison_results(self, df):
        """
        Groups the DataFrame by 'Method', 'SemanticMap', and 'SemanticMapSize',
        aggregating 'ComparisonResult' by summing.
        """
        return df.groupby(['Method', 'SemanticMap', 'SemanticMapSize']).agg({
            "ComparisonResult": "sum"
        }).reset_index()

    def _compute_metric_rates(self, df):
        """
        Computes the top-k rates from 'ComparisonResult' and adds them as new columns.
        """
        df['top_1'] = df['ComparisonResult'].apply(
            lambda x: x.get_top_1_rate() * 100)
        df['top_2'] = df['ComparisonResult'].apply(
            lambda x: x.get_top_2_rate() * 100)
        df['top_3'] = df['ComparisonResult'].apply(
            lambda x: x.get_top_3_rate() * 100)
        df['top_any'] = df['ComparisonResult'].apply(
            lambda x: x.get_top_any_rate() * 100)
        return df

    def _prepare_plotting_data(self, df, metric):
        """
        Prepares the DataFrame for plotting by selecting the desired metric and creating labels.
        """
        # Select the desired metric for plotting
        df['MetricValue'] = df[metric]

        # Create labels combining SemanticMap and Size
        df['SemanticMapLabel'] = df.apply(
            lambda row: f"{row['SemanticMap']} ({row['SemanticMapSize']})", axis=1
        )
        return df

    def _plot_chart(self, df, metric):
        """
        Plots the chart using matplotlib based on the prepared DataFrame.

        Parameters:
        - df (DataFrame): The DataFrame prepared for plotting.
        - metric (str): The performance metric to plot.
        """
        plt.figure(figsize=(12, 6))

        # Define the methods to plot and their corresponding colors and offsets
        method_colors = {
            METHOD_BASE: 'blue',
            METHOD_SELF_REFLECTION: 'orange',
            METHOD_MULTIAGENT_REFLECTION: 'green',
            METHOD_ENSEMBLING: 'red',
        }
        offsets = {
            METHOD_BASE: -0.3,
            METHOD_SELF_REFLECTION: -0.15,
            METHOD_MULTIAGENT_REFLECTION: 0.15,
            METHOD_ENSEMBLING: 0.3,
        }

        # Store method data for trend lines
        method_data = {}

        # Plot data points for each method
        for method in constants.METHODS:
            method_df = df[df['Method'] == method]
            if not method_df.empty:
                x_coords = method_df['SemanticMapSize'] + offsets[method]
                y_coords = method_df['MetricValue']
                plt.scatter(
                    x_coords,
                    y_coords,
                    marker='o',
                    label=method,
                    color=method_colors.get(method)
                )

                # Store data for trend line calculation
                method_data[method] = {
                    'SemanticMapSize': method_df['SemanticMapSize'].values,
                    'MetricValue': method_df['MetricValue'].values,
                    'Color': method_colors.get(method)
                }

        # Plot trend lines for methods with sufficient data
        self._plot_trend_lines(method_data)

        # Customize x-axis labels
        self._customize_x_axis(df)

        # Set chart titles and labels
        plt.xlabel('Semantic Complexity (Number of Objects)')
        plt.ylabel(f"{constants.pretty_metric_constant(metric)} Rate (%)")
        plt.title(f"Average {constants.pretty_metric_constant(
            metric)} vs Semantic Map Complexity")

        # Add legend and grid
        plt.legend()
        plt.grid(True)

        # Adjust layout and display the chart
        plt.tight_layout()
        plt.show()

    def _plot_trend_lines(self, method_data):
        """
        Calculates and plots trend lines for each method using polynomial fitting.

        Parameters:
        - method_data (dict): Dictionary containing data for each method.
        """
        # Compute the overall min and max SemanticMapSize for trend lines
        x_min = min(data['SemanticMapSize'].min()
                    for data in method_data.values())
        x_max = max(data['SemanticMapSize'].max()
                    for data in method_data.values())
        x_trend = np.linspace(x_min, x_max, 100)

        # Calculate and plot the trend lines for methods with sufficient data
        for method, data in method_data.items():
            if len(data['SemanticMapSize']) > 1:
                # Fit a polynomial curve (degree 3)
                coefficients = np.polyfit(
                    data['SemanticMapSize'], data['MetricValue'], 3)
                polynomial = np.poly1d(coefficients)
                y_trend = polynomial(x_trend)

                # Plot the trend line
                plt.plot(
                    x_trend,
                    y_trend,
                    linestyle='--',
                    linewidth=2,
                    color=data['Color'],
                    label=f'{method} Trendline'
                )

    def _customize_x_axis(self, df):
        """
        Customizes the x-axis labels to show semantic map names and sizes.

        Parameters:
        - df (DataFrame): The DataFrame containing the plotting data.
        """
        # Group by SemanticMapSize and aggregate SemanticMap names
        size_to_maps = df.groupby('SemanticMapSize')[
            'SemanticMap'].unique().reset_index()
        size_to_maps['SemanticMapLabel'] = size_to_maps.apply(
            lambda row: ' ,\n'.join(row['SemanticMap']) + f" ({row['SemanticMapSize']})", axis=1
        )

        # Set x-ticks and labels
        x_ticks = size_to_maps['SemanticMapSize']
        x_labels = size_to_maps['SemanticMapLabel']
        plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
