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




def by_files(file_paths):



    consolidated_df = []

    for file_path in file_paths:

        df, filtered_df = by_file(file_path)

        consolidated_df.append({
            "fname": file_path.split('/')[-1],
            "Avg JCT": df['jct'].mean(),
            "p99 JCT": df['jct'].quantile(0.99),
            "p999 JCT": df['jct'].quantile(0.999),
            "Avg Pred Error": filtered_df['error'].mean(),
            "p99 Pred Error": filtered_df['error'].quantile(0.99),
        })


    consolidated_df = pd.DataFrame(consolidated_df)

    consolidated_df["Norm. Avg JCT"] = consolidated_df["Avg JCT"] / consolidated_df["Avg JCT"].min()
    consolidated_df["Norm. p99 JCT"] = consolidated_df["p99 JCT"] / consolidated_df["p99 JCT"].min()
    consolidated_df["Norm. p999 JCT"] = consolidated_df["p999 JCT"] / consolidated_df["p999 JCT"].min()


    # Convert DataFrame to list of lists
    table_data = consolidated_df.values.tolist()

    # Use column names as headers
    headers = consolidated_df.columns

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


    # # plot cdf jcts
    # plt.figure()
    # for policy in pd_data:

    #     df = pd_data[policy].get('df')

    #     # Sort values and compute CDF
    #     df_sorted = df['jct'].sort_values().reset_index(drop=True)
    #     cdf = np.arange(1, len(df_sorted) + 1) / len(df_sorted)

    #     sns.lineplot(x=df_sorted, y=cdf, label=policy)
    #     plt.xscale("log")
    #     plt.xlabel('JCT')
    #     plt.ylabel('CDF')
    # plt.savefig(f"JCT_{trace}.png", format="png", dpi=300)


def by_file(file_path):
    if os.path.exists(file_path):

        df = pd.read_csv(file_path)

        df['jct'] =  df['end_time'] - df['submit_time']
        valid_prediction = (df['estimated_end_time'] == -1) & (df['estimated_start_time'] == -1)
        # valid_prediction = df['estimated_end_time'] != -1

        filtered_df = df.drop(df[valid_prediction].index)

        filtered_df['pred_jct'] =  filtered_df['estimated_end_time'] - filtered_df['submit_time']
        filtered_df['jct'] =  filtered_df['end_time'] - filtered_df['submit_time']

        filtered_df['error'] = 100.0 * (filtered_df['pred_jct'] - filtered_df['jct']).abs() / filtered_df['pred_jct']


        return df, filtered_df
    else:
        print(f"{file_path} doesn't exist")






if __name__ == '__main__':

    valid_policies = ["FIFO", "FS", "BOOST", "SRSF", "THEMIS", "AFS", "PCS_jct", "PCS_bal", "PCS_pred"]
    
    parser = argparse.ArgumentParser()
    

    # Create a mutually exclusive group for the two modes
    mode_group = parser.add_mutually_exclusive_group(required=True)

    # Mode 1: Policies and traces (default policy if not specified)
    mode_group.add_argument(
        "-policies", nargs="+", help="Space-separated list of policies.", choices=valid_policies, default=["*"], type=str
    )
    mode_group.add_argument(
        "-traces", nargs="+", help="Space-separated list of traces (e.g., gavel 0e4a51 etc.)", type=str
    )

    # Mode 2: Only file names
    mode_group.add_argument(
        "-files", nargs="+", help="Space-separated list of file names.", type=str
    )
    args = parser.parse_args()



    if args.policies == ["*"]:
        args.policies = valid_policies

    # Now you can check which mode was specified
    if args.traces:
        print("Mode 1 selected: Using policies and traces.")
        print("Policies:", args.policies)
        print("Traces:", args.traces)
        main(args)

    elif args.files:
        print("Mode 2 selected: Using file names only.")
        print("Files:", args.files)
        by_files(args.files)
