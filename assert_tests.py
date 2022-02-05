import json
from os.path import exists

def missing_field_outer(field,refer_string):
    print(f'Please ensure experiment json file has field "{field}".{refer_string}')
    exit()

def invalid_instance_outer(field,correct_instance,refer_string):
    print(f'"{field}" field of experiment json should be a {correct_instance}.{refer_string}')
    exit()

def missing_field_test(test_num,field,refer_string):
    print(f'Test #{test_num} is missing required field "{field}".{refer_string}')
    exit()

def invalid_instance_test(test_num,field,correct_instance,refer_string):
    print(f'"{field}" field of test #{test_num} should be a {correct_instance}.{refer_string}')
    exit()    


def validate_int_list(int_list,test_num,field,refer_string):
    for potential_int in int_list:
        if not isinstance(potential_int,int):
            invalid_instance_test(test_num,field,"list of ints",refer_string)

def validate_methods(methods_list,test_num,all_methods,refer_string):
    valid_methods = {
        "frequent-directions",
        "zha-simon",
        "bcg"
    }

    test_methods = set()
    for method in methods_list:
        if not isinstance(method,str):
            invalid_instance_test(test_num,"methods","list of strings",refer_string)
        if method not in valid_methods:
            print(f'{method} from test #{test_num} is not a supported method, valid methods are "frequent-directions","zha-simon", and "bcg".\n{refer_string}')
            exit()
        else:
            all_methods.add(method)
            test_methods.add(method)
    return all_methods,test_methods


def validate_tests(tests,refer_string):
    all_methods = set()
    datasets = set()
    for i,test in enumerate(tests):

        test_num = i+1

        if not isinstance(test,dict):
            print(f'test #{test_num} must be a dict.{refer_string}')
            exit()

        if "dataset" not in test:
            missing_field_test(test_num,"dataset",refer_string)
        if not isinstance(test["dataset"],str):
            invalid_instance_test(test_num,"dataset","str",refer_string)
        datasets.add(test["dataset"])

        if "methods" not in test:
            missing_field_test(test_num,"dataset",refer_string)
        if not isinstance(test["methods"],list):
            invalid_instance_test(test_num,"dataset","list",refer_string)
        all_methods,test_methods = validate_methods(test["methods"],test_num,all_methods,refer_string)

        if "m_percent" not in test:
            missing_field_test(test_num,"m_percent",refer_string)
        if not isinstance(test["m_percent"],float):
            invalid_instance_test(test_num,"m_percent","float",refer_string)

        if "n_batches" not in test:
            missing_field_test(test_num,"n_batches",refer_string)
        if not isinstance(test["n_batches"],list):
            invalid_instance_test(test_num,"n_batches","list",refer_string)
        validate_int_list(test["n_batches"],test_num,"n_batches",refer_string)

        if "phis_to_plot" not in test:
            missing_field_test(test_num,"phis_to_plot",refer_string)
        if not isinstance(test["phis_to_plot"],list):
            invalid_instance_test(test_num,"phis_to_plot","list",refer_string)
        validate_int_list(test["phis_to_plot"],test_num,"phis_to_plot",refer_string)  

        if "k_dims" not in test:
            missing_field_test(test_num,"k_dims",refer_string)
        if not isinstance(test["k_dims"],list):
            invalid_instance_test(test_num,"k_dims","list",refer_string)
        validate_int_list(test["k_dims"],test_num,"k_dims",refer_string)   

        if "make_plots" not in test:
            missing_field_test(test_num,"make_plots",refer_string)
        if not isinstance(test["make_plots"],bool):
            invalid_instance_test(test_num,"make_plots","bool",refer_string) 

        if 'bcg' in test_methods:
            if "r_values" not in test:
                missing_field_test(test_num,"r_values since it uses the bcg method",refer_string)
            if not isinstance(test["r_values"],list):
                invalid_instance_test(test_num,"r_values","list",refer_string)
            validate_int_list(test["r_values"],test_num,"r_values",refer_string)
                        
            if 'lam_coeff' not in test:
                missing_field_test(test_num,"lam_coeff since it uses the bcg method",refer_string)
            if not isinstance(test['lam_coeff'],float):
                invalid_instance_test(test_num,"lam_coeff","float",refer_string)

            if 'num_runs' not in test:
                missing_field_test(test_num,"num_runs since it uses the bcg method",refer_string)
            if not isinstance(test['num_runs'],int):
                invalid_instance_test(test_num,"num_runs","int",refer_string)      
    return all_methods,datasets  


def validate_dataset_info(datasets,req_datasets):
    specified_datasets = set()
    for req_dataset in req_datasets:
        if req_dataset not in datasets:
            print(f"Please provide the path to the {req_dataset} dataset or remove it from tests")   
            exit()     
        if not exists(datasets[req_dataset]):
            print(f'Path to numpy file for {req_dataset} dataset does not exist.')
            exit()
        if '.npy' not in datasets[req_dataset]:
            print(f'Please supply {req_dataset} dataset as a numpy file')
            exit()


        
def validate_method_label(methods,req_methods):
    for req_method in req_methods:
        if req_method not in methods:
            print(f"Please provide a method label for the {req_method} method or remove it from tests")
            exit()        
        if not isinstance(methods[req_method],str):
            print(f'Method label for {req_method} must be a string')
            exit()


def validate_experiment(experiment_path):
    # Open JSON file for experiment specification
    f = open(experiment_path)
    try:
        test_spec = json.load(f)
    except ValueError as err:
        print("Tests file is not a valid JSON file. Please double check syntax.")
        exit()
    f.close()

    refer_string = " Please refer to the documentation provided in the repository or look at one of the example experiments under the experiments folder for more help."

    if "tests" not in test_spec:
        missing_field_outer("tests",refer_string)
    if not isinstance(test_spec['tests'],list):
        invalid_instance_outer("tests","list",refer_string)
    methods,datasets = validate_tests(test_spec['tests'],refer_string)

    if "dataset_info" not in test_spec:
        missing_field_outer("dataset_info",refer_string)
    if not isinstance(test_spec['dataset_info'],dict):
        invalid_instance_outer("dataset_info","dict",refer_string)
    validate_dataset_info(test_spec['dataset_info'],datasets)

    if "method_label" not in test_spec:
        missing_field_outer("method_label",refer_string)
    if not isinstance(test_spec['method_label'],dict):
        invalid_instance_outer("method_label","dict",refer_string)
    validate_method_label(test_spec['method_label'],methods)

    return test_spec