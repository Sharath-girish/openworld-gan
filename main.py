import os
import config
import shutil
from stages import *
from config import Config

# List of stages to be executed in the corresponding order
stage_names = ['classifier','generate_features', 'ood_detection', 'clustering', 
               'merge', 'refine']
stage_classes = [StageClassifier, StageGenerateFeatures, StageOOD, StageClustering, 
                 StageMerge, StageRefine]

def clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path,exist_ok=False)

def main(conf):
    # Obtain number of iterations and the final stage of last iteration
    conf_common = config.get_conf_common(conf)
    num_steps = conf_common['num_iters']
    final_stage = conf_common['final_stage']
    assert num_steps>=1
    assert final_stage in stage_names

    ### Each stage saves a config.yaml in its respective root path when the stage completes
    ### save_conf keeps track of config upto current stage of execution.
    ### If pipeline is being resumed, it skips execution of stages with saved config matching
    ### config.yaml in the save path of the stage.
    save_conf = {} 
    root_path = conf_common["save_path"]
    for step in range(1,num_steps+1):
        save_conf_step = {}
        step_save_path = os.path.join(root_path,f'step{step}') # Save path for current iteration
        os.makedirs(step_save_path,exist_ok=True)
        conf_step = config.get_conf_step(conf,step=step)
        num_stages = stage_names.index(final_stage)+1 if step==num_steps else len(stage_classes)
        for stage in range(num_stages):
            stage_name = stage_names[stage]
            save_conf_step[stage_name] = conf_step[stage_name] 
            save_conf[f'step{step}'] = save_conf_step # Update the save_conf for current stage
            stage_save_path = os.path.join(step_save_path,stage_name) # Save path for current stage
            conf_save_path = os.path.join(stage_save_path,'config.yaml') # Path to dumped save_conf
            if conf_common['resume']:
                # Check if there is a saved config in the save path for the current stage
                # If it matches the current save_conf, skip rerunning the stage
                if os.path.exists(conf_save_path):
                    load_conf = config._load_from_file(conf_save_path)
                    if load_conf == save_conf:
                        print(f'Iteration {step} stage {stage_name}'+\
                                  f' completed at {stage_save_path}')
                        continue
            # Clear contents of stage path if rerunning stage, except if it is a classifier
            # training stage, in which case it resumes from the most recent checkpoint saved.
            if not (stage_name in ['classifier','refine'] and conf_common['resume'] \
                    and not os.path.exists(conf_save_path)):
                clear_dir(stage_save_path) 
            if not os.path.exists(stage_save_path):
                os.makedirs(stage_save_path,exist_ok=False)
            conf_common['resume'] = False
            print_str = f'              Iteration {step}: Running stage {stage_name}                '
            print('-'*len(print_str))
            print(print_str)
            print('-'*len(print_str))
            # Create the stage object and execute it
            cur_stage = stage_classes[stage](step, conf, root_path, save_conf)
            cur_stage.execute()
        # Saving the config upto the current iteration in the iteration save path
        config._save_to_file(conf_step,os.path.join(step_save_path,'config.yaml'))
        save_conf[f'step{step}'] = save_conf_step
    # Saving the full config in the root path
    config._save_to_file(save_conf,os.path.join(root_path,'config.yaml'))
            

# Create config dictionary from yaml file to keep track of hyperparameters across stages and iterations
if __name__ == "__main__":
    conf = Config(use_args=True)
    main(conf)
