import numpy as np
from sim import gen_workload
from models import Models
from scipy.stats import linregress

def main():

    

    app_list = {}
    event_queue = list()
    workload = "themis1"
    load=0.8
    num_apps=5000
    num_gpus = 1
    seed = 4567
    models = Models('linear')


    gen_workload(workload,
                workload,
                workload,
                workload,
                load,
                num_gpus,
                num_apps,
                seed,
                app_list,
                event_queue,
                models)


    sizes = [a.service for a in app_list.values()]

    mu = 1.0/np.mean(sizes)
    lmbda = load*mu


    print()

    print(mu)

    print(lmbda)

    decay = mu - lmbda

    print(decay)




if __name__ == '__main__':
    main()