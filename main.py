# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:47:49 2024

@author: amine
"""

import forecast.learner as lea
import forecast.wrangler as wra
from forecast import utilities as util
import sys

env_vars = util.get_env_variables()

def assemble_arguments(task: str = None, kwargs: dict = None):
    
    # Default arguments for a particular task
    if isinstance(task, str):
        match task.lower():
            case 'train':
                kwargs = dict(
                    wrangler_kwargs=dict(
                        dataset=util.read_csv(env_vars.DATA_DIR, env_vars.DATA_FILE),
                        target='carbon_intensity_avg',
                        sort_by='timestamp',
                        drop=['datetime', 'timestamp', 'zone_name', 
                              'production_sources'],
                        shift=-1,
                        test_size=0.2,
                        random_state=42
                        ),
                    learner_kwargs=dict()
                    )
            case 'serve':
                kwargs = dict(
                    wrangler_kwargs=dict(
                        dataset=util.read_csv(env_vars.DATA_DIR, env_vars.DATA_FILE),
                        target='carbon_intensity_avg',
                        sort_by='timestamp'),                    
                    learner=None,
                    )
            case _:
                assert False, f"Wrong task = `{task}`" 
        util.disp(f"Assembling the default arguments for the task = `{task}`.")
    
    # Specific arguments (regardless of the task)
    elif isinstance(kwargs, str):
        try:
            kwargs = eval(kwargs)
            util.disp(
                "The argument string is directly evaluated as a dictionary.")
        except BaseException as e:
            kwargs = util.load_json_as_dict(kwargs)
            util.disp(
                "The argument string is interpreted as a JSON dictionary.")
    elif isinstance(kwargs, dict):
        pass
    else:
        raise f"Invalid combination of task `{task}` and kwargs `{kwargs}`"
    
    assert isinstance(kwargs, dict)
    
    return kwargs

def main(task: str, kwargs = None) -> object:

    kwargs = assemble_arguments(task, kwargs)
    
    match task.lower():
        case 'train':
            util.disp(f"Train the learner.")
            wrangler = wra.Wrangler(**kwargs['wrangler_kwargs'])
            wrangler()
            learner = lea.Learner(
                data=wrangler,
                model_kwargs=kwargs['learner_kwargs'])
            learner()
            learner.explore()
            learner.train()
            learner.test()
            return learner
        case 'serve':
            util.disp(f"Serve the learner.")
            return None
        case _:
            assert False, f"There is no such task  as task = `{task}`."

if __name__ == "__main__":
    
    match len(sys.argv):
        case 2:
            output = main(sys.argv[1])
        case 3:
            kwargs = assemble_arguments(sys.argv[1], sys.argv[2])
            output = main(sys.argv[1], **kwargs)
        case _:
            output = None

#%%
    output = main('train')
    print(f">>> {output.data.target}")
    print(output.data.dataset.train.head())
    dataset = output.data.dataset
    data_test = output.data.dataset.test
    report = output.report