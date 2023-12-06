from parser import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
import operator


def main(args):
    fnames = args.fnames

    # colors = ['b','r','g','k','y','c','darkviolet'] + ['gainsboro','lightcoral','peru','forestgreen','lightgreen','paleturquoise','steelblue','lightsteelblue','blue','mediumorchid','plum','lightpink']

    results = {}

    for fname in fnames:
        results[fname] = parse_file_data(fname)

        if "FS" in fname:
            print(fname)
            fair_result_dictionary = results[fname]

    for fname in fnames:
        perc = 95

        print(".................")
        print(".................")
        print(f"fname: {fname}")
        print(f"avg JCT: {avg_ACT(results[fname])}")
        print(f"p{90} JCT: {np.percentile(cdf_ACT(results[fname])[0], q=90)}")
        print(f"p{95} JCT: {np.percentile(cdf_ACT(results[fname])[0], q=95)}")
        print(f"p{99} JCT: {np.percentile(cdf_ACT(results[fname])[0], q=99)}")
        print(f"avg pred error: {avg_ACT_error(results[fname])}")
        print(
            f"p{90} pred error: {np.percentile(cdf_ACT_error(results[fname])[0], q=90)}"
        )
        print(
            f"p{95} pred error: {np.percentile(cdf_ACT_error(results[fname])[0], q=95)}"
        )
        print(
            f"p{99} pred error: {np.percentile(cdf_ACT_error(results[fname])[0], q=99)}"
        )

        # print(f"max unfairness: {cdf_unfairness(results[fname], fair_result_dictionary)[0][-1]}")
        # print(f"avg unfairness: {avg_unfairness(results[fname], fair_result_dictionary)}")
        # print(f"avg prediction error: {avg_ACT_error(results[fname])}")
        print(".................")
        print(".................")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fnames", nargs="+", help="input filenames", type=str, required=True
    )

    args = parser.parse_args()
    main(args)
