import matplotlib.pyplot as plt

from constants import (
    METHOD_BASE,
    METHOD_ENSEMBLING,
    METHOD_MULTIAGENT_REFLECTION,
    METHOD_SELF_REFLECTION,
)


class ChartNumberObjectsImpactGenerator:

    def __init__(self, df_comparison_results, semantic_map_basenames, semantic_map_sizes, llm_label="Google_gemini-1.5-pro", mode="certainty"):
        self.df = df_comparison_results.copy()
        self.semantic_map_basenames = semantic_map_basenames
        self.semantic_map_sizes = semantic_map_sizes
        self.semantic_map_size_dict = dict(
            zip(self.semantic_map_basenames, self.semantic_map_sizes))
        self.llm_label = llm_label
        self.mode = mode

    def generate_chart(self, metric='top_1'):
        # Filter the DataFrame for the specified LLM and mode
        df_filtered = self.df[
            (self.df['LLM'] == self.llm_label) &
            (self.df['Mode'] == self.mode)
        ]

        # Map semantic map basenames to sizes
        df_filtered['SemanticMapSize'] = df_filtered['SemanticMap'].map(
            self.semantic_map_size_dict)

        print(df_filtered.to_string())

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

        # Plot the data
        plt.figure(figsize=(12, 6))

        methods = [METHOD_BASE, METHOD_SELF_REFLECTION,
                   METHOD_MULTIAGENT_REFLECTION, METHOD_ENSEMBLING]

        for method in methods:
            method_df = df_grouped[df_grouped['Method'] == method]
            if not method_df.empty:
                x_coords = method_df['SemanticMapSize']
                y_coords = method_df['MetricValue']
                plt.scatter(x_coords, y_coords, marker='o', label=method)

        # Get unique SemanticMapSizes and their corresponding labels
        unique_sizes_labels = df_grouped[[
            'SemanticMapSize', 'SemanticMapLabel']].drop_duplicates().sort_values('SemanticMapSize')

        x_ticks = unique_sizes_labels['SemanticMapSize']
        x_labels = unique_sizes_labels['SemanticMapLabel']

        plt.xlabel('Semantic Map Size (Number of Objects)')
        plt.ylabel(f'{metric.replace("_", " ").capitalize()} Rate (%)')
        plt.title('Comparison Result vs. Semantic Map Size')

        # Set x-ticks at the semantic map sizes with custom labels
        plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
