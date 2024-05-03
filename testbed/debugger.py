import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import argparse, operator

def file_data_filter(fdata, col_idx, comparison_op, compare_value, delimiter=','):
    
    filtered_fdata = list()

    for entry in fdata:
        if comparison_op(float(entry.rstrip().split(delimiter)[col_idx]), compare_value):
            filtered_fdata.append(entry)
    return filtered_fdata


def timelines(y, timeline, color=['cornflowerblue']):
    """Plot timelines at y from xstart to xstop with given color."""   
    
    submit = timeline[0]
    dispatch=timeline[1]
    start = timeline[2]
    end = timeline[3]

    # plt.hlines(y, submit, start, colors[0], lw=20)
    # plt.hlines(y+0.5, start, end, color, lw=33.5)

    ax.add_patch(Rectangle((dispatch,y),(start-dispatch),1, fill=False,color=color,hatch='//'))
    ax.add_patch(Rectangle((start,y),(end-start),1,fill=True, color=color))



    # plt.vlines(xstart, y+0.04, y-0.04, 'g', lw=2)
    # plt.vlines(xstop, y+0.05, y-0.05, color, lw=2)


# on a per job basis
def get_free_gpu_index(job_submit_time, job_dispatch_time, job_start_time, job_end_time, gpu_free_time):
    index = None

    # earliest free sorting
    # gpu_free_time = sorted(gpu_free_time, key=lambda k: k[1])


    for gpu in gpu_free_time:
        if gpu[1] <= job_dispatch_time:
            index = gpu[0]

            
            # slack for queued jobs
            if job_submit_time <= gpu[1]:
                slack.append(job_start_time - gpu[1])
            

            gpu[1] = job_end_time
            return index

    print("ERROR")
    sys.exit(1)


def place_app(app, gpu_free_time):
    
    '''
    start_times = list(map(lambda t: t[2], app["job_data"]))
    end_times = list(map(lambda t: t[3], app["job_data"]))

    app["submit_time"] = app["job_data"][0][0]
    app["dispatch_time"] = app["job_data"][0][1]
    app["start_time"] = min(start_times)
    app["end_time"] = max(end_times)
    '''


    annotation_x = ((app["start_time"] + app["end_time"])/2)
    annotation_y = None


    job_data = sorted(app["job_data"], key=lambda k: k[3], reverse=True)

    for jd in job_data:
        submit_time, dispatch_time, start_time, end_time = jd
        y = get_free_gpu_index(submit_time, dispatch_time, start_time, end_time, gpu_free_time)
        timelines(y, [submit_time, dispatch_time, start_time, end_time], app["color"])

        plt.annotate(str(int(start_time)), (start_time, y), fontsize=4)
        plt.annotate(str(int(end_time)), (end_time-5, y), fontsize=4)


        if annotation_y == None:
            annotation_y = y

    plt.annotate(app["app_id"], (annotation_x,annotation_y+0.5))


def parse_file_data(fname, vis_apps, delimiter=','):

    result_dictionary = {}

    with open(fname) as fd:
        keys, *file_data = fd.readlines()

    file_data = file_data_filter(file_data, 0, operator.ge, min(vis_apps))
    file_data = file_data_filter(file_data, 0, operator.le, max(vis_apps))

    keys = keys.split(delimiter)
    for i, key in enumerate(keys):
        result_dictionary[key] = list(map(lambda e: float(e.split(delimiter)[i]), file_data))

    return result_dictionary



def cdf_duration(result_dictionary, filter=None):
    duration = np.subtract(result_dictionary["end_time"], result_dictionary["start_time"])
    duration.sort()
    cdf = np.linspace(1.0/len(duration),1.0,len(duration))
    return duration, cdf
    


def cdf_setup_time(result_dictionary, filter=None):
    setup_time = np.subtract(result_dictionary["start_time"], result_dictionary["dispatch_time"])
    setup_time.sort()
    cdf = np.linspace(1.0/len(setup_time),1.0,len(setup_time))
    return setup_time, cdf



def cdf_queuing_delay(result_dictionary, filter=None):
    queuing_delay = np.subtract(result_dictionary["app_start_time"], result_dictionary["app_submit_time"])
    queuing_delay.sort()
    cdf = np.linspace(1.0/len(queuing_delay),1.0,len(queuing_delay))
    return queuing_delay, cdf
    


def cdf_app_exec(result_dictionary, filter=None):
    app_exec = np.subtract(result_dictionary["app_end_time"], result_dictionary["app_start_time"])
    app_exec.sort()
    cdf = np.linspace(1.0/len(app_exec),1.0,len(app_exec))
    return app_exec, cdf
    


if __name__ == '__main__':


    

    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', nargs="+", help = "input filenames", type=str)
    args = parser.parse_args()


    fnames = args.fnames


    starting_app_id = np.random.choice(range(1500))


    vis_apps = [0,1,2,3,4,5,6,7]
    vis_apps = list(range(597,604))


    print(fnames)

    clr = ['gainsboro','lightcoral','peru','forestgreen','lightgreen','paleturquoise','steelblue','lightsteelblue','blue','mediumorchid','plum','lightpink']
    
    results = {}
    for fname in fnames:
        results[fname] = parse_file_data(fname, vis_apps)






    for fname in fnames:

        plt.figure()
        ax = plt.gca()

        print(ax)


        gpu_free_time = [[0,0], [1,0], [2,0], [3,0], [4,0], [5,0], [6,0], [7,0]]

        slack = list()

        apps = {}
        result_dictionary = results[fname]
        label = fname.split('.')[0]







        result_dictionary["app_id"] = list(map(int, result_dictionary["app_id"]))

        min_time = min(result_dictionary["submit_time"] + result_dictionary["dispatch_time"] + result_dictionary["start_time"] + result_dictionary["end_time"])
        

        
        result_dictionary["submit_time"] = list(map(lambda t: t - min_time, result_dictionary["submit_time"]))
        result_dictionary["dispatch_time"] = list(map(lambda t: t - min_time, result_dictionary["dispatch_time"]))
        result_dictionary["start_time"] = list(map(lambda t: t - min_time, result_dictionary["start_time"]))
        result_dictionary["end_time"] = list(map(lambda t: t - min_time, result_dictionary["end_time"]))
        

        for i, app_id in enumerate(result_dictionary["app_id"]):

            if app_id not in apps:
                apps[app_id] = {"color": clr[app_id % len(clr)], "job_data": list(), "app_id": str(app_id)}
            apps[app_id]["job_data"].append([result_dictionary["submit_time"][i],
                                            result_dictionary["dispatch_time"][i],
                                            result_dictionary["start_time"][i],
                                            result_dictionary["end_time"][i]])



        for app_id in apps:
            app = apps[app_id]

            start_times = list(map(lambda t: t[2], app["job_data"]))
            end_times = list(map(lambda t: t[3], app["job_data"]))

            app["submit_time"] = app["job_data"][0][0]
            app["dispatch_time"] = app["job_data"][0][1]
            app["start_time"] = min(start_times)
            app["end_time"] = max(end_times)


        result_dictionary["app_submit_time"] = [apps[app_id]["submit_time"] for app_id in apps]
        result_dictionary["app_start_time"] = [apps[app_id]["start_time"] for app_id in apps]
        result_dictionary["app_end_time"] = [apps[app_id]["end_time"] for app_id in apps]

        xticks = []


        

        for app_id in vis_apps:
            place_app(apps[app_id], gpu_free_time)
            xticks += [apps[app_id]["end_time"]] + [apps[app_id]["start_time"]]

        xticks.sort()
        xticks = list(map(int, xticks))

        plt.rcParams['hatch.linewidth'] = 5
        plt.xlim(min(xticks),max(xticks))
        plt.ylim(0, 8)
        plt.xticks(xticks, fontsize=3)
        plt.xlabel("Time (sec)")
        plt.ylabel('GPU')
        plt.grid(alpha=.3, linestyle='--')
        plt.savefig('%s_vis.png' % (label), dpi = 300)



    colors = ['b','r','g','k','y','c','darkviolet']

    for func in [cdf_duration, cdf_setup_time, cdf_queuing_delay, cdf_app_exec]:
        plt.figure()
        for i, fname in enumerate(fnames):


            result_dictionary = results[fname]
            x,y = func(result_dictionary)

            label = fname.split('.')[0]
            plt.plot(x, y, color=colors[i], marker='o', markevery=10, label=label)

        plt.legend(loc="best")
        plt.ylim(0, 1.0)
        plt.xlabel("Time")
        plt.ylabel('CDF')
        plt.grid(alpha=.3, linestyle='--')
        plt.savefig('%s.png' % str(func.__name__), dpi = 300)




    # plt.figure()
    # for entry in data:
    #     app_id,job_id,num_gpus,submit_time,start_time,end_time,theoretical_duration,actual_duration,dispatch_time = entry.rstrip().split(',')

    #     app_id = int(app_id)
    #     job_id = int(job_id)
    #     submit_time = float(submit_time)
    #     start_time = float(dispatch_time)
    #     end_time = float(end_time)
    #     theoretical_duration = float(theoretical_duration)
    #     actual_duration = float(actual_duration)





    #     if app_id not in apps:
    #         apps[app_id] = {"color": clr[app_id % len(clr)], "job_data": list(), "app_id": str(app_id)}
    #     apps[app_id]["job_data"].append([submit_time, start_time, end_time])



    # # print(np.mean(slack))

    # '''
    # delta = list()

    # for app_id in apps:
    #     start_times = apps[app_id]
    #     min_start_time = min(start_times)
    #     max_start_time = max(start_times)

    #     delta.append(max_start_time - min_start_time)



    # cdf = np.linspace(1.0/len(delta),1.0,len(delta))
    # delta.sort()
    # '''

    # # plt.plot(delta, cdf, color='b', marker='o', markevery=10)
    # plt.ylim(0, 8)
    # plt.xticks(xticks, fontsize=3)
    # plt.xlabel("Time (sec)")
    # plt.ylabel('GPU')
    # plt.grid(alpha=.3, linestyle='--')
    # plt.savefig('%s_vis.png' % (args.result_file.split('.')[0]), dpi = 300)



