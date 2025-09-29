import pandas as pd
import glob

exp_id = "test_run"

result_files = glob.glob(f"{exp_id}/fan_origin_batch-*.tsv")
df_list = [pd.read_csv(f, sep='\t') for f in result_files]
results_df = pd.concat(df_list, ignore_index=True)

results_df['response'] = pd.to_numeric(results_df['response'], errors='coerce')

print(results_df.head(5))
