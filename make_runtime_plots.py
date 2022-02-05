from truncatedSVD.utils import init_figure
from truncatedSVD.utils import check_and_create_dir


def make_plots(specs_json,cache_dir):
    f = open(specs_json)
    try:
        spec = json.load(f)
    except ValueError as err:
        print("Tests file is not a valid JSON file. Please double check syntax.")
        exit()
    f.close()
    datasets = listdir(cache_dir)

    for dataset in datasets:

        valid_datasets = spec['datasets']
        if dataset not in valid_datasets:
            print(f"{dataset} is not valid. Skipping this dataset.")
            continue
        
        print(f'Generating plot for dataset: {dataset}')

################## Generate plots showing runtimes for different methods across diferent ranks but same number of updates ###################        
        for n_batches in spec['n_batches']:
            empty = True
            #pdb.set_trace()
            print(f'\tGenerating for batch split: {n_batches}')
            fig, ax = init_figure(f'Runtimes for {n_batches} Updates','Rank','Runtime (ms)')
            
            for method in listdir(normpath(join(cache_dir, dataset))):
                #pdb.set_trace()
                if method not in spec['method_label'].keys():
                    continue
                print(f'\t\tGenerating for method: {spec["method_label"][method]}')
                runtime_dict = {}

                for experiment_dir in listdir(normpath(join(cache_dir,dataset,method))):
                    if f'n_batches_{n_batches}' in experiment_dir:
                        k_dim = int(experiment_dir.split("_")[-1])

                        if method != "bcg":
                            max_seen_runtime_phi = 0        
                            max_file = ""                    
                            for file in listdir(normpath(join(cache_dir,dataset,method,experiment_dir))):
                                if "runtime" in file:
                                    phi_num = int(file.split("_")[2].split('.')[0])
                                    if phi_num > max_seen_runtime_phi:
                                        max_seen_runtime_phi = phi_num
                                        max_file = file

                            with open(normpath(join(cache_dir,dataset,method,experiment_dir,max_file)),'rb') as f:
                                runtime = float(np.load(f)) 

                            runtime_dict[k_dim] = runtime
                        else:
                            runtime_sum = 0
                            num_runs = 0
                            for run_dir in listdir(normpath(join(cache_dir,dataset,method,experiment_dir))):
                                if "run" not in  run_dir:
                                    continue                                
                                max_seen_runtime_phi = 0        
                                max_file = ""                    
                                for file in listdir(normpath(join(cache_dir,dataset,method,experiment_dir,run_dir))):
                                    if "runtime" in file:
                                        phi_num = int(file.split("_")[2].split('.')[0])
                                        if phi_num > max_seen_runtime_phi:
                                            max_seen_runtime_phi = phi_num
                                            max_file = file

                                with open(normpath(join(cache_dir,dataset,method,experiment_dir,run_dir,max_file)),'rb') as f:
                                    runtime_sum += float(np.load(f)) 
                                num_runs += 1

                            runtime_dict[k_dim] = runtime_sum/num_runs                            

                sorted_runtimes = sorted(runtime_dict.items())
                if len(sorted_runtimes) != 0:
                    empty = False
                runtime_ranks = [x[0] for x in sorted_runtimes]
                runtime_vals = [x[1] for x in sorted_runtimes]
                ax.plot(runtime_ranks,runtime_vals,label=spec['method_label'][method],marker='o')


            ax.legend(loc="upper right")

            plt.figure(fig.number)
            if not empty:
                check_and_create_dir(join(cache_dir,dataset,'runtime_figures'))
                check_and_create_dir(join(cache_dir,dataset,'runtime_figures','runtime_vs_rank'))                
                plt.savefig(
                    normpath(join(cache_dir, dataset,'runtime_figures','runtime_vs_rank',f"{dataset}_runtimes_batch_split_{n_batches}")),
                    bbox_inches="tight",
                    pad_inches=0.2,
                    dpi=200,
                )
            else:
                print("No results for this set of experiments exist for this dataset, skipping")
            plt.close()        


################## Generate plots showing runtimes for different methods across diferent number of updates but same rank ###################        
        for ranks in spec['ranks']:
            empty = True
            print(f'\tGenerating for Rank: {ranks}')
            fig,ax = init_figure(f'Runtimes For {ranks} Rank','Number of Updates','Runtime (ms)')
            for method in listdir(normpath(join(cache_dir,dataset))):
                #pdb.set_trace()
                if method not in spec['method_label'].keys():
                    continue
                print(f'\t\tGenerating for method: {spec["method_label"][method]}')
                runtime_dict = {}

                for experiment_dir in listdir(normpath(join(cache_dir,dataset,method))):

                    if f'k_dims_{ranks}' in experiment_dir:
                        batch_split = int(experiment_dir.split("_")[3])

                        if method != "bcg":
                            max_seen_runtime_phi = 0        
                            max_file = ""                    
                            for file in listdir(normpath(join(cache_dir,dataset,method,experiment_dir))):
                                if "runtime" in file:
                                    phi_num = int(file.split("_")[2].split('.')[0])
                                    if phi_num > max_seen_runtime_phi:
                                        max_seen_runtime_phi = phi_num
                                        max_file = file

                            with open(normpath(join(cache_dir,dataset,method,experiment_dir,max_file)),'rb') as f:
                                runtime = float(np.load(f)) 

                            runtime_dict[batch_split] = runtime
                        else:
                            runtime_sum = 0
                            num_runs = 0
                            for run_dir in listdir(normpath(join(cache_dir,dataset,method,experiment_dir))):
                                if "run" not in  run_dir:
                                    continue                                
                                max_seen_runtime_phi = 0        
                                max_file = ""                    
                                for file in listdir(normpath(join(cache_dir,dataset,method,experiment_dir,run_dir))):
                                    if "runtime" in file:
                                        phi_num = int(file.split("_")[2].split('.')[0])
                                        if phi_num > max_seen_runtime_phi:
                                            max_seen_runtime_phi = phi_num
                                            max_file = file

                                with open(normpath(join(cache_dir,dataset,method,experiment_dir,run_dir,max_file)),'rb') as f:
                                    runtime_sum += float(np.load(f)) 
                                num_runs += 1

                            runtime_dict[batch_split] = runtime_sum/num_runs                            

                sorted_runtimes = sorted(runtime_dict.items())
                if len(sorted_runtimes) != 0:
                    empty = False                
                runtime_batch_split = [x[0] for x in sorted_runtimes]
                runtime_vals = [x[1] for x in sorted_runtimes]
                ax.plot(runtime_batch_split,runtime_vals,label=spec['method_label'][method],marker='o')


            ax.legend(loc="upper right")

            plt.figure(fig.number)
            if not empty:
                check_and_create_dir(join(cache_dir,dataset,'runtime_figures'))
                check_and_create_dir(join(cache_dir,dataset,'runtime_figures','runtime_vs_num_updates'))
                plt.savefig(
                    normpath(join(cache_dir, dataset,'runtime_figures','runtime_vs_num_updates',f"{dataset}_runtimes_k_dim_{ranks}")),
                    bbox_inches="tight",
                    pad_inches=0.2,
                    dpi=200,
                )
            else:
                print("No results for this set of experiments exist for this dataset, skipping")
            plt.close()        


if __name__ == "__main__":
    
    import argparse
    import json
    from os import listdir
    from os.path import normpath,join
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)    

    arg_parser = argparse.ArgumentParser(description="Create plots for comparing covariance errors across methods for specified numbers of updates and k dimension")
    
    arg_parser.add_argument(
        "--cache_dir",
        "-c",
        dest="cache_dir",
        required=True,
        help="Specifies path to cache folder generated by run_tests.py that contains all the covariance error data"
    )
    arg_parser.add_argument(
        "--specs_json",
        "-s",
        dest="specs_json",
        required=True,
        help="Path to json file containing list of tuples specifying which number of updates to create plots for"
    )

    args = arg_parser.parse_args()
    
    make_plots(args.specs_json, args.cache_dir)