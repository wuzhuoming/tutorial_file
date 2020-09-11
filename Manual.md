# **Manual for CUHKPrototypeTuner**

## Install CUHKPrototypeTuner

1. Run the following command 
    ```bash
    pip install nni && \ 
    wget https://github.com/vincentcheny/hpo-training/releases/download/v1.2/CUHKPrototypeTuner-1.2-py3-none-any.whl && \ 
    nnictl package install CUHKPrototypeTuner-1.2-py3-none-any.whl
    ```
2. if success install, you should see this output  in the command line

    ```bash
    Processing ./CUHKPrototypeTuner-1.2-py3-none-any.whl
    Installing collected packages: CUHKPrototypeTuner
    Successfully installed CUHKPrototypeTuner-1.2
    CUHKPrototypeTuner installed!
    ```

### For ELMO 
1. Clone the source code of ELMO from git 
    ```bash
    git clone https://github.com/allenai/bilm-tf.git
    ```

1. Go to the working dir
    ```bash
    cd bilm-tf
    ```
   
2. Create file  "search_space.json" to define the search space of hyperparmaeters and hardware parameters. Execute: 
    ```bash
    cat << EOF > search_space.json
    {
        "epoch":{"_type": "uniform", "_value": [5, 50]},
        "batch_size":{"_type": "uniform", "_value": [64, 512]},
        "optimizer":{"_type":"choice","_value":["Chris: please add optimizer here also"]},
        
        "inter_op_parallelism_threads":{"_type":"choice","_value":[1,2,3,4]},
        "intra_op_parallelism_threads":{"_type":"choice","_value":[2,4,6,8,10,12]},
        "infer_shapes":{"_type":"choice","_value":[0,1]},
        "place_pruned_graph":{"_type":"choice","_value":[0,1]},
        "enable_bfloat16_sendrecv":{"_type":"choice","_value":[0,1]},
        "do_common_subexpression_elimination":{"_type":"choice","_value":[0,1]},
        "max_folded_constant":{"_type":"choice","_value":[2,4,6,8,10]},
        "do_function_inlining":{"_type":"choice","_value":[0,1]},
        "global_jit_level":{"_type":"choice","_value":[0,1,2]},
        "tf_gpu_thread_mode":{"_type":"choice","_value":["global", "gpu_private", "gpu_shared"]}
    }
    EOF
    ```

3. Create file  "config.yml" with following content. Execute:
    ```bash
    cat << EOF > config.yml
    authorName: lscm
    experimentName: elmo
    trialConcurrency: 1
    
    # Specific the maximum runing time, we specific 1 week here
    maxExecDuration: 40h 
    maxTrialNum: 9999
    trainingServicePlatform: local
    searchSpacePath: search_space.json
    useAnnotation: false
    tuner:
      builtinTunerName: CUHKPrototypeTuner
    trial:
      command: python ./bin/train_elmo.py --train_prefix=./data/one_billion/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* --vocab_file ./data/vocab-2016-09-10.txt --save_dir ./save_output
      codeDir: .
    
      # We assume there is 8 gpu(s)
      gpuNum: 8
    EOF
    ```

4. Update the following files for applying configuration from tuner and reporting performance metrics

| File  | Download |
| ------------- | ------------- |
| bilm/training.py | [Link]()  |
| bin/train_elmo.py | [Link]()  |

<!--
Chris: Please find upload the modified version to github
For the details of the modification please check [here](Manual_elmo.md)
-->

### For mBART (training program: mBART)  
Chris: ZM please fill it

### For mBART (training program: mBART)  
Chris: ZM please fill it

## Start training

1. Inside the working directory, execute 
    ```bash
    nnictl create --config ./config.yml -p 8080
    ```

2. If successfully start the tuner with training program, you should see the following message in the console:
Your experiment id `egchD4qy` is shown. Please note it down for [pausing the tuning](#Stop And Resume tuning) later on. 

    ```log
    INFO: Starting restful server...
    INFO: Successfully started Restful server!
    INFO: Setting local config...
    INFO: Successfully set local config!
    INFO: Starting experiment...
    INFO: Successfully started experiment!
    -----------------------------------------------------------------------
    The experiment id is egchD4qy
    The Web UI urls are: [Your IP]:8080
    -----------------------------------------------------------------------
    
    You can use these commands to get more information about the experiment
    -----------------------------------------------------------------------
             commands                       description
    1. nnictl experiment show        show the information of experiments
    2. nnictl trial ls               list all of trial jobs
    3. nnictl top                    monitor the status of running experiments
    4. nnictl log stderr             show stderr log content
    5. nnictl log stdout             show stdout log content
    6. nnictl stop                   stop an experiment
    7. nnictl trial kill             kill a trial job by id
    8. nnictl --help                 get help information about nnictl
    -----------------------------------------------------------------------
    ```

4. To check Tuning Process From Web UI, open a browser and use the URL given by NNI:

```
The Web UI urls are: [Your IP]:8080
```

#### **Overview page**

Information about this experiment will be shown in the WebUI, including the experiment trial profile and search space message. NNI also supports downloading this information and the parameters through the **Download** button. You can download the experiment results during or after the execution.

<img src="https://lh3.googleusercontent.com/-LzY00M6Qj5s/X1r3WeZybeI/AAAAAAAAAdc/rDSdoBefp5oDhz3XKMHBxbTL3zr1ENHcACK8BGAsYHg/s0/2020-09-10.png"/>

The top 10 trials will be listed on the Overview page. You can browse all the trials on the “Trials Detail” page.

<img src="https://lh3.googleusercontent.com/-nUV6DIuojdw/X1r3Fzve2TI/AAAAAAAAAdU/SYO259Bld24Lzhvsn7UfJu8XT7NGnpOBQCK8BGAsYHg/s0/2020-09-10.png"/>

#### **Trials Details page**

Click the “Default Metric” tab to see the point graph of all trials. Hover to see specific default metrics and search space messages.

<img src="https://lh3.googleusercontent.com/-U9ZPPo3mY1s/X1r4Uu9cLYI/AAAAAAAAAdk/GCutOTm5FDQlLJ-HVyriXuphdU2BJRwtgCK8BGAsYHg/s0/2020-09-10.png"/>

Click the “Hyper Parameter” tab to see the parallel graph.

- You can select the percentage to see the top trials.

- Choose two axis to swap their positions.

  <img src="https://nni.readthedocs.io/en/latest/_images/hyperPara.png"/>

  

  Below is the status of all trials. Specifically:

  - Trial detail: trial’s id, duration, status, accuracy, and search space file.

  - If you run on the OpenPAI platform, you can also see the hdfsLogPath.

  - Kill: you can kill a job that has the `Running` status.

  - Support: Used to search for a specific trial.

    <img src="https://lh3.googleusercontent.com/-zGe44tn04GY/X1r4y8KRZRI/AAAAAAAAAds/0VCQ0B3eQbcbKMM323WQdKGz9xPQTJrdgCK8BGAsYHg/s0/2020-09-10.png" />



## Stop And Resume tuning

To stop current experiment, you can use this command:

```
nnictl stop egchD4qy # egchD4qy is the experiment id provided by the tuner when you launch it 
```

To resume the experiment you stop:

```
nnictl resume egchD4qy # egchD4qy is the experiment id provided by the tuner when you launch it 
```

Get the result