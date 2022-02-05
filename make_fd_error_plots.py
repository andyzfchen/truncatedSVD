import pdb
from truncatedSVD.utils import init_figure


def make_plots(specs_json, cache_dir):
    f = open(specs_json)
    try:
        spec = json.load(f)
    except ValueError as err:
        print(f"{specs_json} is not a valid JSON file. Please double check syntax.")
        exit()
    f.close()
    
    datasets = listdir(cache_dir)

    for dataset in datasets:

        valid_datasets = spec['datasets']
        if dataset not in valid_datasets:
            continue

        print(f'Generating for dataset: {dataset}')
        for num_updates in spec['n_batches']:
            cov_empty = proj_empty = True
            print(f'\tGenerating for batch split: {num_updates}')
            cov_fig,cov_ax = init_figure(f'Covariance Error For {num_updates} Updates','Rank','Covariance Error')
            proj_fig,proj_ax = init_figure(f'Projection Error For {num_updates} Updates','Rank','Projection Error')
            for method in listdir(normpath(join(cache_dir,dataset))):
                #pdb.set_trace()
                if method not in spec['method_label'].keys():
                    continue
                print(f'\t\tGenerating for method: {spec["method_label"][method]}')
                cov_errs_dict = {}
                proj_errs_dict = {}

                for dir in listdir(normpath(join(cache_dir,dataset,method))):

                    if f'n_batches_{num_updates}' in dir:
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
                                if "run" not in  run_dir:
                                    continue
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
                if len(sorted_cov) != 0:
                    cov_empty = False
                cov_ranks = [x[0] for x in sorted_cov]
                cov_vals = [x[1] for x in sorted_cov]
                cov_ax.plot(cov_ranks,cov_vals,label=spec['method_label'][method],marker='o')

                sorted_proj = sorted(proj_errs_dict.items())
                if len(sorted_proj) != 0:
                    proj_empty = False
                proj_ranks = [x[0] for x in sorted_proj]
                proj_vals = [x[1] for x in sorted_proj]
                proj_ax.plot(proj_ranks,proj_vals,label=spec['method_label'][method],marker='o')

            cov_ax.legend(loc="upper right")
            proj_ax.legend(loc="upper right")

            plt.figure(cov_fig.number)
            if not cov_empty:
                plt.savefig(
                    normpath(join(cache_dir, dataset,f"{dataset}_covariance_errors_batch_split_{num_updates}")),
                    bbox_inches="tight",
                    pad_inches=0.2,
                    dpi=200,
                )
            else:
                print("No results for this set of experiments exists, skipping")
            plt.close()

            plt.figure(proj_fig.number)
            if not proj_empty:
                plt.savefig(
                    normpath(join(cache_dir, dataset,f"{dataset}_projection_errors_batch_split_{num_updates}")),
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
        "--specs_json",
        "-s",
        dest="specs_json",
        required=True,
        help="Path to json file containing list of tuples specifying which number of updates to create plots for"
    )
    arg_parser.add_argument(
        "--cache_dir",
        "-c",
        dest="cache_dir",
        required=True,
        help="Specifies path to cache folder generated by run_tests.py that contains all the covariance error data"
    )

    args = arg_parser.parse_args()
    
    make_plots(args.specs_json, args.cache_dir)