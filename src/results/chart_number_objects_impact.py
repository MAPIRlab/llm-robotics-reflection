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

        # Group by Method and SemanticMap, aggregating ComparisonResults
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

        # Pivot the DataFrame to have methods as columns
        df_pivot = df_grouped.pivot_table(
            index='SemanticMapSize', columns='Method', values='MetricValue')

        # Sort the DataFrame by SemanticMapSize
        df_pivot = df_pivot.sort_index()

        # Plot the data
        plt.figure(figsize=(10, 6))

        methods = [METHOD_BASE, METHOD_SELF_REFLECTION,
                   METHOD_MULTIAGENT_REFLECTION, METHOD_ENSEMBLING]

        for method in methods:
            if method in df_pivot.columns:
                plt.plot(df_pivot.index,
                         df_pivot[method], marker='o', label=method)

        plt.xlabel('Semantic Map Size (Number of Objects)')
        plt.ylabel(f'{metric.replace("_", " ").capitalize()} Rate (%)')
        plt.title('ComparisonResult vs. Semantic Map Size')
        plt.legend()
        plt.grid(True)
        plt.show()
