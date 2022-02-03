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

        if dataset == "figures":
            continue

        print(f'Generating for dataset: {dataset}')
        for num_updates in spec['num_updates']:
            print(f'\tGenerating for batch split: {num_updates}')
            cov_fig,cov_ax = create_figure(f'Covariance Error For {num_updates} Updates','Rank','Covariance Error')
            proj_fig,proj_ax = create_figure(f'Projection Error For {num_updates} Updates','Rank','Projection Error')
            for method in listdir(normpath(join(cache_dir,dataset))):
                #pdb.set_trace()
                if method not in spec['method_label'].keys():
                    continue
                print(f'\t\tGenerating for method: {spec["method_label"][method]}')
                cov_errs_dict = {}
                proj_errs_dict = {}

                for dir in listdir(normpath(join(cache_dir,dataset,method))):

                    if f'batch_split_{num_updates}' in dir:
                        k_dim = int(dir.split("_")[-1])

                        if method != "bcg":
                            with open(normpath(join(cache_dir,dataset,method,dir,"covarience_error.pkl")),'rb') as f:
                                covariance_errs = pickle.load(f) 

                            with open(normpath(join(cache_dir,dataset,method,dir,"projection_error.pkl")),'rb') as f:
                                proj_errs = pickle.load(f)

                            cov_errs_dict[k_dim] = covariance_errs[next(reversed(covariance_errs))]
                            proj_errs_dict[k_dim] = proj_errs[next(reversed(proj_errs))]
                        else:
                            cov_err_sum = 0
                            proj_err_sum = 0
                            num_runs = 0
                            for run_dir in listdir(normpath(join(cache_dir,dataset,method,dir))):
                                with open(normpath(join(cache_dir,dataset,method,dir,run_dir,"covarience_error.pkl")),'rb') as f:
                                    covariance_errs = pickle.load(f) 

                                with open(normpath(join(cache_dir,dataset,method,dir,run_dir,"projection_error.pkl")),'rb') as f:
                                    proj_errs = pickle.load(f)

                                cov_err_sum += covariance_errs[next(reversed(covariance_errs))]
                                proj_err_sum += proj_errs[next(reversed(proj_errs))]
                                num_runs += 1

                            cov_errs_dict[k_dim] = cov_err_sum/num_runs
                            proj_errs_dict[k_dim] = proj_err_sum/num_runs                              

                sorted_cov = sorted(cov_errs_dict.items())
                cov_ranks = [x[0] for x in sorted_cov]
                cov_vals = [x[1] for x in sorted_cov]
                cov_ax.plot(cov_ranks,cov_vals,label=spec['method_label'][method])

                sorted_proj = sorted(proj_errs_dict.items())
                proj_ranks = [x[0] for x in sorted_proj]
                proj_vals = [x[1] for x in sorted_proj]
                proj_ax.plot(proj_ranks,proj_vals,label=spec['method_label'][method])

            cov_ax.legend(loc="upper right")
            proj_ax.legend(loc="upper right")

            plt.figure(cov_fig.number)
            plt.savefig(
                normpath(join(cache_dir, dataset,f"covariance_errors_batch_split_{num_updates}")),
                bbox_inches="tight",
                pad_inches=0.2,
                dpi=200,
            )
            plt.close()

            plt.figure(proj_fig.number)
            plt.savefig(
                normpath(join(cache_dir, dataset,f"projection_errors_batch_split_{num_updates}")),
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