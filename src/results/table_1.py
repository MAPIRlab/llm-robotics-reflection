import pandas as pd


class Table1Generator:

    def __init__(self, df_comparison_results, mode: str, llm_label: str = "Google_gemini-1.5-pro"):
        self.df = df_comparison_results.copy()
        self.mode = mode
        self.llm_label = llm_label
        self.table_df = None  # Will hold the final DataFrame

    def generate_table(self):

        # Filter by "LLM" and "Mode"
        self.filter_mode_llm()

        # HERE! for each Mode, Method, LLM and SemanticMap add a row called "Average", by aggregating the ComparisonResult values from all the Queries
        self.add_average_per_mode_method_llm()

        # Add "Dataset" column
        self.add_dataset_column()

        # Add "Query Type" columns
        self.add_query_type_column()

        print(self.table_df)

        # Group by "Dataset", "Method" and "Query Type" columns, aggregating ComparisonResults
        self.group_by_dataset_method_and_query_type()

        print(self.table_df.to_string())

        # Add "top_1", "top_2", "top_3", "top_any" columns from ComparisonResults
        self.add_top_k_columns()

        # Change "top_1", "top_2", "top_3", "top_any" to differences w.r.t. "Method" = base
        self.compute_top_k_differences()

        # Sort and rearrange columns
        self.finalize_table()

        return self.table_df

    def add_average_per_mode_method_llm(self):
        # Group by Mode, Method, LLM, and SemanticMap
        sum_rows = self.table_df.groupby(
            ['Mode', 'Method', 'LLM', 'SemanticMap'], as_index=False
        ).agg({"ComparisonResult": "sum"})

        # Add a column to denote this is an "Average" row
        sum_rows['QueryID'] = 'Average'

        # Append the summed rows to the original DataFrame
        self.table_df = pd.concat([self.table_df, sum_rows], ignore_index=True)

    def filter_mode_llm(self):
        self.table_df = self.df[
            (self.df['Mode'] == self.mode) &
            (self.df['LLM'] == self.llm_label)
        ]

    def _map_dataset(self, semantic_map):
        if semantic_map.startswith("scannet_"):
            return "scannet"
        elif semantic_map.startswith("scenenn_"):
            return "scenenn"
        return "unknown"  # Default if none match

    def _map_query_type(self, query_id):

        # Average row
        if query_id == "Average":
            return "Average"

        # Extract the numeric part of the QueryID
        query_number = int(query_id.split('_')[-1])

        # Map the QueryID to the Query Type
        if 1 <= query_number <= 10:
            return "Descriptive"
        elif 11 <= query_number <= 20:
            return "Affordance"
        elif 21 <= query_number <= 30:
            return "Negation"
        return "Unknown"  # Default if out of range

    def add_dataset_column(self):
        self.table_df["Dataset"] = self.table_df['SemanticMap'].apply(
            self._map_dataset)

    def add_query_type_column(self):
        self.table_df["Query Type"] = self.table_df['QueryID'].apply(
            self._map_query_type)

    def group_by_dataset_method_and_query_type(self):
        self.table_df = self.table_df.groupby(
            ['Dataset', 'Method', 'Query Type'], as_index=False
        ).agg({"ComparisonResult": "sum"})

    def add_top_k_columns(self):
        self.table_df['top_1'] = self.table_df['ComparisonResult'].apply(
            lambda x: x.get_top_1_rate())
        self.table_df['top_2'] = self.table_df['ComparisonResult'].apply(
            lambda x: x.get_top_2_rate())
        self.table_df['top_3'] = self.table_df['ComparisonResult'].apply(
            lambda x: x.get_top_3_rate())
        self.table_df['top_any'] = self.table_df['ComparisonResult'].apply(
            lambda x: x.get_top_any_rate())
        self.table_df = self.table_df.drop(columns=['ComparisonResult'])

    def merge_base_values(self):
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

    def calculate_differences(self):
        for col in ['top_1', 'top_2', 'top_3', 'top_any']:
            base_col = f'base_{col}'
            diff_col = f'diff_{col}'
            self.table_df[diff_col] = self.table_df[col] - \
                self.table_df[base_col]

    def format_metrics(self):
        def format_value(row, col):
            if row['Method'] == 'base':
                return f"{row[col]:.2f}"
            else:
                diff = row[f'diff_{col}']
                if diff > 0:
                    return f"+{diff:.2f}"
                else:
                    return f"{diff:.2f}"
        for col in ['top_1', 'top_2', 'top_3', 'top_any']:
            self.table_df[col] = self.table_df.apply(
                lambda row: format_value(row, col), axis=1)

    def cleanup_dataframe(self):
        cols_to_drop = [f'base_{col}' for col in ['top_1', 'top_2', 'top_3', 'top_any']] + \
                       [f'diff_{col}' for col in [
                           'top_1', 'top_2', 'top_3', 'top_any']]
        self.table_df = self.table_df.drop(columns=cols_to_drop)

    def compute_top_k_differences(self):
        # Merge base method values
        self.merge_base_values()

        # Calculate differences from the base method
        self.calculate_differences()

        # Format the metric values for display
        self.format_metrics()

        # Clean up the DataFrame
        self.cleanup_dataframe()

    def finalize_table(self):
        self.table_df = self.table_df[[
            'Dataset', 'Method', 'Query Type', 'top_1', 'top_2', 'top_3', 'top_any']]
        self.table_df = self.table_df.sort_values(
            by=['Dataset', 'Method', 'Query Type'])
