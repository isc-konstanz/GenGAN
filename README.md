# Motivation

This project aims to fulfill five tasks:

1. To provide a structured framework in which timeseries GAN's can be **implemented**, **evaluated** 
and **compared** such that performance of all three requires from the user singularly the definition of a 
model and the provision of the appropriate data for this model.

2. To provide a number of timeseries GAN models implemented in tensorflow which can be passed through the framework, to
be implemented, evaluated and compared to other models previously passed through the framework.

3. To provide a number of methods which can be chosen by the user to be performed during the evaluation step
when a timeseries GAN model is passed through the framework.

4. To provide a number of methods which can be chosen by the user to be performed during the comparison step,
which provide a sensible comparison of the current evaluation of the timeseries GANs performance to previous
evaluations already performed.

5. To allow the user to start the framework at any step of the frameworks task list (Implementation, Evaluation, 
Comparison) provided that the required information for the completion of this step is available.


## What is a "Framework"?
I want to be specific about what I mean when I say the word "Framework" so that it is clear what this
repo attempts to accomplish. To answer this question we must take a closer look at what it means to complete
the tasks of Implementation, Evaluation, and Comparison, once this has been done we can make a more descriptive
definition of what is meant by the word "Framework".

The tasks Implementation, Evaluation, and Completion each consist of a number of subtasks, which each require 
a number of inputs, whose completion produces a number of outputs. Furthermore, the completion of a singular 
subtask is reliant upon the output of other subtasks. To complicate things even further, the factorization of 
each of the main tasks into a set of sensible subtasks is in many respects an arbitrary choice. With this in 
mind I can now explain a little more concretely what I mean by the word "Framework".

My "Framework", present in this repo, factors the three main subtasks each into a set of 'sensible' subtasks.
Furthermore, it defines the order by which each of these subtasks are to be completed as well as the method
by which each subtask finds the appropriate inputs and produces the appropriate outputs. Lastly, it defines
which of the outputs, produced in this whole process, are relevant to the user and stores them in the appropriate
locations.

## File Tree
```
├───bin
│   └───train.py
├───conf
│   ├───base_model.cfg
│   └───model_a.cfg
├───databi
│   ├───database_1
│   └───database_2
│       ├───sample_day_1.csv
│       ├───sample_day_2.csv
│           
├───evaluation
│   └───batch_plot
│       ├───el_power
│       └───pv_power
│           ├───GAN_day1_plt.jpg
│           ├───GAN_day2_plt.jpg
│           
├───evaluators
│   └───evaluate.py
├───hist
│   └───05_11_2021_ex
│       ├───model_a
│       └───model_b
│           ├───conf
│           ├───evaluation
│           └───results
├───lib
│   ├───parse_confs.py
│   ├───prep_data.py
├───models
│   ├───model_a.py
│   └───model_b.py
└───results
    ├───saved_model
```

## Framework Definitions and Conventions

There are a number of conventions that are implicitely and explicitely assumed by the framework. So that the user
doesn't run into issues by inadvertently violating one of the implicit conventions defined by this framework they
will all be listed here. A number of these conventions can be deduced from the file tree.

1. The data present in the database folder should be timeseries data stored in .csv files with no holes or gaps; 
In other words the time difference between consecutive steps of the time-series should be constant. If there is
more then one file in data_dir, it is assumed that they are indexed such that os.listdir iterates through them 
in time_sequential order. It is further assumed that the time difference between the first step of one file and 
the first step of the preceding file is equal to the time difference between consecutive steps within a singular 
file.

2. The results dir contains solely the highest order representation of any run of the framework from which the
rest of the run can be exactly replicated with identical config files, that being the saved trained model.

3. Each method present in the module evaluate.py which the user would like to run during the evaluation step of
frameworks workflow should have its corresponding key, present in the models config file under the section
'Evaluation', set to true.

4. All methods in the module evaluate.py works on batch level data. That being generated data of the shape
(batch_size, steps_per_sample, features_per_sample)

5. Each object in the evaluation dir is the output of a singular method in the module evaluate.py

## Workflow of the Framework
In order to understand how to use this application, it is naturally important to understand what will happen if
you were to use it. Here we will briefly describe the basic tasks that will be accomplished when the train.py
module is run as '__main__', what outputs will arise, where they will be saved, and what the user can do with these
outputs once the module has finished running. WIP
