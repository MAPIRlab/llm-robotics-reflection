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

    def __init__(self, df_comparison_results, semantic_map_basenames, semantic_map_sizes, llm_label="Google_gemini-1.5-pro", mode="certainty"):
        self.df = df_comparison_results.copy()
        self.semantic_map_basenames = semantic_map_basenames
        self.semantic_map_sizes = semantic_map_sizes
        self.semantic_map_size_dict = dict(
            zip(self.semantic_map_basenames, self.semantic_map_sizes))
        self.llm_label = llm_label
        self.mode = mode

    def _pretty_metric(self, metric: str):
        if metric == "top_1":
            return "Top-1"
        elif metric == "top_2":
            return "Top-2"
        elif metric == "top_3":
            return "Top-3"
        elif metric == "top_any":
            return "Top-Any"

    def generate_chart(self, metric='top_any'):
        # Filter the DataFrame for the specified LLM and mode
        df_filtered = self.df[
            (self.df['LLM'] == self.llm_label) &
            (self.df['Mode'] == self.mode)
        ]

        # Map semantic map basenames to sizes
        df_filtered['SemanticMapSize'] = df_filtered['SemanticMap'].map(
            self.semantic_map_size_dict)

        # Group by Method, SemanticMap, and SemanticMapSize
        df_grouped = df_filtered.groupby(['Method', 'SemanticMap', 'SemanticMapSize']).agg({
            "ComparisonResult": "sum"
        }).reset_index()

        # Compute the desired metric rates
        df_grouped['top_1'] = df_grouped['ComparisonResult'].apply(
            lambda x: x.get_top_1_rate() * 100)
        df_grouped['top_2'] = df_grouped['ComparisonResult'].apply(
            lambda x: x.get_top_2_rate() * 100)
        df_grouped['top_3'] = df_grouped['ComparisonResult'].apply(
            lambda x: x.get_top_3_rate() * 100)
        df_grouped['top_any'] = df_grouped['ComparisonResult'].apply(
            lambda x: x.get_top_any_rate() * 100)

        # Select the desired metric
        df_grouped['MetricValue'] = df_grouped[metric]

        # Create labels combining SemanticMap and Size
        df_grouped['SemanticMapLabel'] = df_grouped.apply(
            lambda row: f"{row['SemanticMap']} ({row['SemanticMapSize']})", axis=1)

        plt.figure(figsize=(12, 6))

        methods = [METHOD_BASE, METHOD_SELF_REFLECTION,
                   METHOD_MULTIAGENT_REFLECTION, METHOD_ENSEMBLING]

        # Define offsets for each method
        offsets = {
            constants.METHOD_BASE: -0.3,
            constants.METHOD_SELF_REFLECTION: -0.15,
            constants.METHOD_MULTIAGENT_REFLECTION: 0.15,
            constants.METHOD_ENSEMBLING: 0.3,
        }

        # Update the colors for each method
        method_colors = {
            METHOD_BASE: 'blue',
            METHOD_SELF_REFLECTION: 'orange',
            METHOD_MULTIAGENT_REFLECTION: 'green',
            METHOD_ENSEMBLING: 'red',
        }

        # Dictionaries to store method data
        method_data = {}

        for method in methods:
            method_df = df_grouped[df_grouped['Method'] == method]
            if not method_df.empty:
                x_coords = method_df['SemanticMapSize'] + offsets[method]
                y_coords = method_df['MetricValue']
                plt.scatter(x_coords, y_coords, marker='o',
                            label=method, color=method_colors.get(method))

                # Store the original SemanticMapSize and MetricValue
                method_data[method] = {
                    'SemanticMapSize': method_df['SemanticMapSize'].values,
                    'MetricValue': method_df['MetricValue'].values,
                    'Color': method_colors.get(method)
                }

        # Compute the overall min and max SemanticMapSize for trend lines
        x_min = df_grouped['SemanticMapSize'].min()
        x_max = df_grouped['SemanticMapSize'].max()
        x_trend = np.linspace(x_min, x_max, 100)

        # Calculate and plot the trend lines for all methods
        for method in [METHOD_BASE, METHOD_SELF_REFLECTION, METHOD_MULTIAGENT_REFLECTION]:
            data = method_data.get(method)
            if data and len(data['SemanticMapSize']) > 1:
                # Fit a linear regression line
                coefficients = np.polyfit(
                    data['SemanticMapSize'], data['MetricValue'], 3)
                polynomial = np.poly1d(coefficients)

                # Compute y-values for the trend line over the full x-axis range
                y_trend = polynomial(x_trend)

                # Plot the trend line
                plt.plot(x_trend, y_trend, linestyle='--', linewidth=2,
                         color=data['Color'], label=f'{method} Trendline')

        # Group SemanticMapSizes and combine SemanticMap names for x-axis labels
        size_to_maps = df_grouped.groupby('SemanticMapSize')[
            'SemanticMap'].unique().reset_index()
        size_to_maps['SemanticMapLabel'] = size_to_maps.apply(
            lambda row: ' ,\n'.join(row['SemanticMap']) + f" ({row['SemanticMapSize']})", axis=1)

        x_ticks = size_to_maps['SemanticMapSize']
        x_labels = size_to_maps['SemanticMapLabel']

        plt.xlabel('Semantic Complexity (Number of Objects)')
        plt.ylabel(f'{self._pretty_metric(metric)} Rate (%)')
        plt.title(f"Average {self._pretty_metric(metric)
                             } depending on Semantic Map Complexity")

        # Set x-ticks at the semantic map sizes with custom labels
        plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
