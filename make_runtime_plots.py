import pdb

def create_figure(title,xlabel,ylabel):
        # Initialize figure
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.grid(True, which="both", linewidth=1, linestyle="--", color="k", alpha=0.1)
        ax.tick_params(
            which="both", direction="in", bottom=True, top=True, left=True, right=True
        )

        # Label figure
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")

        return fig,ax



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
            continue
        print(f'Generating for dataset: {dataset}')

################## Generate plots showing runtimes for different methods across diferent ranks but same number of updates ###################        
        for num_updates in spec['num_updates']:
            print(f'\tGenerating for batch split: {num_updates}')
            fig,ax = create_figure(f'Runtimes For {num_updates} Updates','Rank','Runtime (ms)')
            for method in listdir(normpath(join(cache_dir,dataset))):
                #pdb.set_trace()
                if method not in spec['method_label'].keys():
                    continue
                print(f'\t\tGenerating for method: {spec["method_label"][method]}')
                runtime_dict = {}

                for dir in listdir(normpath(join(cache_dir,dataset,method))):

                    if f'batch_split_{num_updates}' in dir:
                        k_dim = int(dir.split("_")[-1])

                        if method != "bcg":
                            max_seen_runtime_phi = 0        
                            max_file = ""                    
                            for file in listdir(normpath(join(cache_dir,dataset,method,dir))):
                                if "runtime" in file:
                                    phi_num = int(file.split("_")[2].split('.')[0])
                                    if phi_num > max_seen_runtime_phi:
                                        max_seen_runtime_phi = phi_num
                                        max_file = file

                            with open(normpath(join(cache_dir,dataset,method,dir,max_file)),'rb') as f:
                                runtime = float(np.load(f)) 

                            runtime_dict[k_dim] = runtime
                        else:
                            runtime_sum = 0
                            num_runs = 0
                            for run_dir in listdir(normpath(join(cache_dir,dataset,method,dir))):
                                max_seen_runtime_phi = 0        
                                max_file = ""                    
                                for file in listdir(normpath(join(cache_dir,dataset,method,dir,run_dir))):
                                    if "runtime" in file:
                                        phi_num = int(file.split("_")[2].split('.')[0])
                                        if phi_num > max_seen_runtime_phi:
                                            max_seen_runtime_phi = phi_num
                                            max_file = file

                                with open(normpath(join(cache_dir,dataset,method,dir,run_dir,max_file)),'rb') as f:
                                    runtime_sum += float(np.load(f)) 
                                num_runs += 1

                            runtime_dict[k_dim] = runtime_sum/num_runs                            

                sorted_runtimes = sorted(runtime_dict.items())
                runtime_ranks = [x[0] for x in sorted_runtimes]
                runtime_vals = [x[1] for x in sorted_runtimes]
                ax.plot(runtime_ranks,runtime_vals,label=spec['method_label'][method],marker='o')


            ax.legend(loc="upper right")

            plt.figure(fig.number)
            plt.savefig(
                normpath(join(cache_dir, dataset,f"runtimes_batch_split_{num_updates}")),
                bbox_inches="tight",
                pad_inches=0.2,
                dpi=200,
            )
            plt.close()        


################## Generate plots showing runtimes for different methods across diferent number of updates but same rank ###################        
        for ranks in spec['ranks']:
            print(f'\tGenerating for Rank: {ranks}')
            fig,ax = create_figure(f'Runtimes For {ranks} Rank','Number of Updates','Runtime (ms)')
            for method in listdir(normpath(join(cache_dir,dataset))):
                #pdb.set_trace()
                if method not in spec['method_label'].keys():
                    continue
                print(f'\t\tGenerating for method: {spec["method_label"][method]}')
                runtime_dict = {}

                for dir in listdir(normpath(join(cache_dir,dataset,method))):

                    if f'k_dims_{ranks}' in dir:
                        batch_split = int(dir.split("_")[3])

                        if method != "bcg":
                            max_seen_runtime_phi = 0        
                            max_file = ""                    
                            for file in listdir(normpath(join(cache_dir,dataset,method,dir))):
                                if "runtime" in file:
                                    phi_num = int(file.split("_")[2].split('.')[0])
                                    if phi_num > max_seen_runtime_phi:
                                        max_seen_runtime_phi = phi_num
                                        max_file = file

                            with open(normpath(join(cache_dir,dataset,method,dir,max_file)),'rb') as f:
                                runtime = float(np.load(f)) 

                            runtime_dict[batch_split] = runtime
                        else:
                            runtime_sum = 0
                            num_runs = 0
                            for run_dir in listdir(normpath(join(cache_dir,dataset,method,dir))):
                                max_seen_runtime_phi = 0        
                                max_file = ""                    
                                for file in listdir(normpath(join(cache_dir,dataset,method,dir,run_dir))):
                                    if "runtime" in file:
                                        phi_num = int(file.split("_")[2].split('.')[0])
                                        if phi_num > max_seen_runtime_phi:
                                            max_seen_runtime_phi = phi_num
                                            max_file = file

                                with open(normpath(join(cache_dir,dataset,method,dir,run_dir,max_file)),'rb') as f:
                                    runtime_sum += float(np.load(f)) 
                                num_runs += 1

                            runtime_dict[batch_split] = runtime_sum/num_runs                            

                sorted_runtimes = sorted(runtime_dict.items())
                runtime_batch_split = [x[0] for x in sorted_runtimes]
                runtime_vals = [x[1] for x in sorted_runtimes]
                ax.plot(runtime_batch_split,runtime_vals,label=spec['method_label'][method],marker='o')


            ax.legend(loc="upper right")

            plt.figure(fig.number)
            plt.savefig(
                normpath(join(cache_dir, dataset,f"runtimes_k_dim_{ranks}")),
                bbox_inches="tight",
                pad_inches=0.2,
                dpi=200,
            )
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