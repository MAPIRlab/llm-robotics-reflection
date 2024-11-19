class Table1Generator:
    def __init__(self, df_comparison_results, mode: str, llm_provider: str = "Google_gemini-1.5-pro"):
        self.df = df_comparison_results.copy()
        self.mode = mode
        self.llm_provider = llm_provider
        self.table_df = None  # Will hold the final DataFrame

    def generate_table(self):
        # Step 1: Filter the DataFrame
        self.filter_data()

        # Step 2: Add 'Dataset' and 'Query Type' columns
        self.add_dataset_and_query_type()

        # Step 3: Aggregate the ComparisonResult objects
        self.aggregate_results()

        # Step 4: Extract metrics from ComparisonResult
        self.extract_metrics()

        # Step 5: Merge base method values
        self.merge_base_values()

        # Step 6: Calculate differences from the base method
        self.calculate_differences()

        # Step 7: Format the metric values for display
        self.format_metrics()

        # Step 8: Clean up the DataFrame
        self.cleanup_dataframe()

        # Step 9: Sort and rearrange columns
        self.finalize_table()

        return self.table_df

    def filter_data(self):
        self.table_df = self.df[
            (self.df['Mode'] == self.mode) &
            (self.df['LLM'] == self.llm_provider)
        ]

    def map_dataset(self, semantic_map):
        if semantic_map.startswith("scannet_scene"):
            return "scannet"
        elif semantic_map.startswith("scenenn_"):
            return "scenenn"
        return "unknown"  # Default if none match

    def map_query_type(self, query_id):
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

    def add_dataset_and_query_type(self):
        self.table_df["Dataset"] = self.table_df['SemanticMap'].apply(
            self.map_dataset)
        self.table_df["Query Type"] = self.table_df['QueryID'].apply(
            self.map_query_type)

    def aggregate_results(self):
        self.table_df = self.table_df.groupby(
            ['Dataset', 'Method', 'Query Type'], as_index=False
        ).agg({"ComparisonResult": "sum"})

    def extract_metrics(self):
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

    def finalize_table(self):
        self.table_df = self.table_df[[
            'Dataset', 'Method', 'Query Type', 'top_1', 'top_2', 'top_3', 'top_any']]
        self.table_df = self.table_df.sort_values(
            by=['Dataset', 'Method', 'Query Type'])
