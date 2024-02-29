# Capacity Management and Learning in Cloud Computing
## How to run  
You can run two types of experiments.  
### Learning Algorithm
Run the following command in the terminal
> python numerical_study.py <param_filename> <months>

For example, the command corresponding to the algorithm with $T = 12\tau$ and $\lambda\sim U[3,6]$ is

> python numerical_study.py data36.pickle 12

### Clairvoyant Problem
The command for solving clairvoyant problems is similar to learning algorithm
> python claivoyant.py <data_filename> <months>

### Analyse the results
To get figures from experiment results, you should run this command
> python draw_std.py

Note that you should <data_filename> and <month_list> in the code file manually.

## Others
* You can find all parameter files in [./params](https://github.com/CyrusNiry/cloudcomputing/tree/main/params).
* Current experiment results are in the folder [./res](https://github.com/CyrusNiry/cloudcomputing/tree/main/res).
