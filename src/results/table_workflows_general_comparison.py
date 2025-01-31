import pandas as pd


class TableWorkflowsGeneralComparisonGenerator:
    """
    Generates a general comparison table for workflows by aggregating over datasets and query types.
    """

    def __init__(self, all_comparison_results_df, mode: str, llm_label: str):
        self.df = all_comparison_results_df.copy()
        self.mode = mode
        self.llm_provider_name = llm_label
        self.table_df = None  # Will hold the processed DataFrame

    def generate_table(self):
        """
        Main method to generate the general comparison table.
        """
        # Step 1: Filter the DataFrame by mode and LLM
        self.filter_by_mode_and_llm()

        # Step 2: Add average rows per Mode, Method, LLM, and SemanticMap
        self.add_average_rows()

        # Step 3: Map 'SemanticMap' to 'Dataset' and add as a new column
        self.add_dataset_column()

        # Step 4: Map 'QueryID' to 'Query Type' and add as a new column
        self.add_query_type_column()

        # Step 5: Group by 'Dataset', 'Method', and 'Query Type', aggregating 'ComparisonResult'
        self.aggregate_comparison_results()

        # Step 6: Extract top-k rates from 'ComparisonResult' and add as new columns
        self.extract_top_k_rates()

        # Step 7: Compute differences in top-k rates relative to the 'base' method
        self.compute_top_k_differences()

        # Step 8: Finalize the table by rearranging and sorting columns
        self.finalize_table()

        # Step 9: Aggregate over datasets and query types to get a general comparison by method
        self.aggregate_by_method()

        return self.table_df

    def filter_by_mode_and_llm(self):
        """
        Filters the DataFrame to include only rows matching the specified mode and LLM label.
        """
        self.table_df = self.df[
            (self.df['Mode'] == self.mode) &
            (self.df['LLM'] == self.llm_provider_name)
        ]

    def add_average_rows(self):
        """
        Adds average rows for each combination of Mode, Method, LLM, and SemanticMap
        by aggregating 'ComparisonResult' over all queries.
        """
        # Group by Mode, Method, LLM, and SemanticMap and sum 'ComparisonResult'
        average_rows = self.table_df.groupby(
            ['Mode', 'Method', 'LLM', 'SemanticMap'], as_index=False
        ).agg({"ComparisonResult": "sum"})

        # Add 'QueryID' column with value 'Average' to indicate these are average rows
        average_rows['QueryID'] = 'Average'

        # Append the average rows to the original DataFrame
        self.table_df = pd.concat(
            [self.table_df, average_rows], ignore_index=True)

    def add_dataset_column(self):
        """
        Adds a 'Dataset' column to the DataFrame by mapping 'SemanticMap' values.
        """
        self.table_df["Dataset"] = self.table_df['SemanticMap'].apply(
            self._map_semantic_map_to_dataset)

    @staticmethod
    def _map_semantic_map_to_dataset(semantic_map: str) -> str:
        """
        Maps a 'SemanticMap' value to a 'Dataset' name.
        """
        if semantic_map.startswith("scannet_"):
            return "scannet"
        elif semantic_map.startswith("scenenn_"):
            return "scenenn"
        else:
            return "unknown"

    def add_query_type_column(self):
        """
        Adds a 'Query Type' column to the DataFrame by mapping 'QueryID' values.
        """
        self.table_df["Query Type"] = self.table_df['QueryID'].apply(
            self._map_query_id_to_query_type)

    @staticmethod
    def _map_query_id_to_query_type(query_id: str) -> str:
        """
        Maps a 'QueryID' to a 'Query Type' based on its numeric part.
        """
        # Handle 'Average' row
        if query_id == "Average":
            return "Average"

        # Extract the numeric part of the QueryID
        try:
            query_number = int(query_id.split('_')[-1])
        except ValueError:
            return "Unknown"

        # Map the query number to a query type
        if 1 <= query_number <= 10:
            return "Descriptive"
        elif 11 <= query_number <= 20:
            return "Affordance"
        elif 21 <= query_number <= 30:
            return "Negation"
        else:
            return "Unknown"

    def aggregate_comparison_results(self):
        """
        Groups the DataFrame by 'Dataset', 'Method', and 'Query Type',
        aggregating 'ComparisonResult' by summing.
        """
        self.table_df = self.table_df.groupby(
            ['Dataset', 'Method', 'Query Type'], as_index=False
        ).agg({"ComparisonResult": "sum"})

    def extract_top_k_rates(self):
        """
        Extracts top-k rates from 'ComparisonResult' and adds them as new columns.
        """
        self.table_df['top_1'] = self.table_df['ComparisonResult'].apply(
            lambda x: x.get_top_1_rate() * 100)
        self.table_df['top_2'] = self.table_df['ComparisonResult'].apply(
            lambda x: x.get_top_2_rate() * 100)
        self.table_df['top_3'] = self.table_df['ComparisonResult'].apply(
            lambda x: x.get_top_3_rate() * 100)
        self.table_df['top_any'] = self.table_df['ComparisonResult'].apply(
            lambda x: x.get_top_any_rate() * 100)

        # Drop the 'ComparisonResult' column as it's no longer needed
        self.table_df = self.table_df.drop(columns=['ComparisonResult'])

    def compute_top_k_differences(self):
        """
        Computes differences in top-k rates relative to the 'base' method and formats the values.
        """
        # Merge base method values into the DataFrame
        self.merge_base_method_values()

        # Calculate differences from the base method
        self.calculate_differences_from_base()

        # Format the metric values for display
        self.format_top_k_values()

        # Clean up temporary columns
        self.cleanup_temporary_columns()

    def merge_base_method_values(self):
        """
        Merges the top-k rates of the 'base' method into the DataFrame for difference calculation.
        """
        base_df = self.table_df[self.table_df['Method'] == 'base']
        base_df = base_df[['Dataset', 'Query Type', 'top_1', 'top_2', 'top_3', 'top_any']].rename(
            columns={
                'top_1': 'base_top_1',
                'top_2': 'base_top_2',
                'top_3': 'base_top_3',
                'top_any': 'base_top_any'
            }
        )
        self.table_df = self.table_df.merge(
            base_df, on=['Dataset', 'Query Type'], how='left')

    def calculate_differences_from_base(self):
        """
        Calculates the differences of top-k rates from the 'base' method.
        """
        for metric in ['top_1', 'top_2', 'top_3', 'top_any']:
            base_metric = f'base_{metric}'
            diff_metric = f'diff_{metric}'
            self.table_df[diff_metric] = self.table_df[metric] - \
                self.table_df[base_metric]

    def format_top_k_values(self):
        """
        Formats the top-k rate values, showing differences from the 'base' method.
        """
        def format_value(row, metric):
            if row['Method'] == 'base':
                # Base method just shows the numeric value
                return f"{row[metric]:.2f}"
            else:
                diff = row[f'diff_{metric}']
                # Add a space after '+' for positive differences
                if diff > 0:
                    return f"+ {diff:.2f}"
                else:
                    return f"{diff:.2f}"

        for metric in ['top_1', 'top_2', 'top_3', 'top_any']:
            self.table_df[metric] = self.table_df.apply(
                lambda row: format_value(row, metric), axis=1
            )

    def cleanup_temporary_columns(self):
        """
        Removes temporary columns used for calculating differences.
        """
        temp_columns = [f'base_{metric}' for metric in ['top_1', 'top_2', 'top_3', 'top_any']] + \
                       [f'diff_{metric}' for metric in [
                           'top_1', 'top_2', 'top_3', 'top_any']]
        self.table_df = self.table_df.drop(columns=temp_columns)

    def finalize_table(self):
        """
        Finalizes the table by rearranging columns and sorting the DataFrame.
        """
        self.table_df = self.table_df[[
            'Dataset', 'Method', 'Query Type', 'top_1', 'top_2', 'top_3', 'top_any'
        ]]
        self.table_df = self.table_df.sort_values(
            by=['Dataset', 'Method', 'Query Type'])

    def aggregate_by_method(self):
        """
        Aggregates over all datasets and query types to produce an overall comparison by Method.
        We convert the formatted strings back to floats, aggregate them, and reformat.
        """
        numeric_columns = ['top_1', 'top_2', 'top_3', 'top_any']

        # Convert the formatted strings (like '+8.00', '-60.00') back to floats for aggregation
        def to_float(val):
            if isinstance(val, str):
                return float(val.replace('+', ''))
            return val

        for col in numeric_columns:
            self.table_df[col] = self.table_df[col].apply(to_float)

        # Group by Method and compute the mean of the metrics
        method_summary = self.table_df.groupby('Method', as_index=False)[
            numeric_columns].mean()

        # Format the values again with two decimal places
        for col in numeric_columns:
            method_summary[col] = method_summary[col].apply(
                lambda x: f"{x:.2f}")

        self.table_df = method_summary[['Method'] + numeric_columns]
