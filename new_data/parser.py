import os
import sys
import pandas as pd
import numpy as np
import argparse
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")  # Or use any other Seaborn style


def main(args):


    for trace in args.traces:

        policy_data = {}
        pd_data = {}

        min_avg_jct = float('inf')
        min_p99_jct = float('inf')
        for policy in args.policies:

            file_path = os.path.join(os.path.dirname(__file__), "{}_{}_result.csv".format(policy, trace))

            if os.path.exists(file_path):

                df = pd.read_csv(file_path)

                df['jct'] =  df['end_time'] - df['submit_time']
                valid_prediction = (df['estimated_end_time'] == -1) & (df['estimated_start_time'] == -1)
                # valid_prediction = df['estimated_end_time'] != -1

                filtered_df = df.drop(df[valid_prediction].index)

                filtered_df['pred_jct'] =  filtered_df['estimated_end_time'] - filtered_df['submit_time']
                filtered_df['jct'] =  filtered_df['end_time'] - filtered_df['submit_time']

                filtered_df['error'] = 100.0 * (filtered_df['pred_jct'] - filtered_df['jct']).abs() / filtered_df['pred_jct']

                pd_data[policy] = {"df": df, "filtered_df": filtered_df}

                min_avg_jct = df['jct'].mean() if min_avg_jct > df['jct'].mean() else min_avg_jct
                min_p99_jct = df['jct'].quantile(0.99) if min_p99_jct > df['jct'].quantile(0.99) else min_p99_jct
            else:
                print(f"{file_path} doesn't exist. Skipping (policy:{policy}, trace:{trace})")


        table_data = []
        header = ["Policy", "Avg JCT", "p99 JCT", "p999 JCT", "Norm. Avg JCT", "Norm. p99 JCT", "Avg Pred Error", "p99 Pred Error",]
        for policy in pd_data:

            df = pd_data[policy].get('df')
            filtered_df = pd_data[policy].get('filtered_df')

            table_data.append([policy,
                df['jct'].mean(),
                df['jct'].quantile(0.99),
                df['jct'].quantile(0.999),
                df['jct'].mean()/min_avg_jct,
                df['jct'].quantile(0.99)/min_p99_jct,
                filtered_df['error'].mean(),
                filtered_df['error'].quantile(0.99),
                ])

        print(f"trace: {trace}")
        print(tabulate(table_data, headers=header, tablefmt='fancy_grid'))
        print()

        # plot cdf jcts
        plt.figure()
        for policy in pd_data:

            df = pd_data[policy].get('df')

            # Sort values and compute CDF
            df_sorted = df['jct'].sort_values().reset_index(drop=True)
            cdf = np.arange(1, len(df_sorted) + 1) / len(df_sorted)

            sns.lineplot(x=df_sorted, y=cdf, label=policy)
            plt.xscale("log")
            plt.xlabel('JCT')
            plt.ylabel('CDF')
        plt.savefig(f"JCT_{trace}.png", format="png", dpi=300)






if __name__ == '__main__':

    valid_policies = ["FIFO", "FS", "SRSF", "THEMIS", "AFS", "PCS_jct", "PCS_bal", "PCS_pred"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-policies", nargs="+", help="space seperated list of policies.", choices= valid_policies, default = ["*"], type=str
    )

    parser.add_argument(
        "-traces", nargs="+", help="space seperated list of traces (e.g., gavel 0e4a51 etc.)", type=str, required=True
    )


    args = parser.parse_args()

    if args.policies == ["*"]:
        args.policies = valid_policies
    main(args)