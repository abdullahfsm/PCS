import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
import operator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', nargs="+", help = "input filenames", type=str)
    args = parser.parse_args()

    fnames = args.fnames

    fname = fnames[0]
    colors = ['b','r','g','k','y','c','darkviolet']
    


    with open(fname) as fp:
        keys, *file_data = fp.readlines()

    x = list(map(lambda e: float(e.rstrip().split(',')[0]), file_data))
    y = list(map(lambda e: float(e.rstrip().split(',')[1]), file_data))

    xkey, ykey = keys.rstrip().split(',')

    print("plotting")

    print("average(y): %f" % np.mean(y))



    plt.plot(x, y, color=colors[0], marker='o', markevery=100)
    plt.xlabel(xkey)
    plt.ylabel(ykey)
    plt.grid(alpha=.3, linestyle='--')
    plt.savefig('%s.png' % (fname.split('.')[0]), dpi = 300)