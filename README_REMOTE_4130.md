# TensorFlow

### Instruction:

#### Step 1: Install [python2.7](https://www.python.org/downloads/) and add path to environment variables.

#### Step 2: Install pip.
    1.Copy [get-pip.py](https://pip.pypa.io/en/stable/installing/) and execute "python get-pip.py".
    2.Add C:\\program\\python\\Script path to environment vairiables.
        Open command line and run "pip list" to test if everything is working."pip list" should ouput list of modules installed on your system.
        
#### Step 3.Install numpy.
    1. Execute "pip install numpy"
    
#### Step 3: Install matplotlib.

    1. Execute "pip install matplotlib"
  
### Execution:
    Before running any script check  __status__ of that file. Status should be "working".
    1. run "python digit_classification.py" or any other script.
    
### Result : 
    1. Check output folder for accuracy value and loss value graph.
    2. Check test_result under results folder.

## Expermentation:
### Tuning parameters in scripts:
    learning_rate = 0.9 - 0.00000001  #The rate at which value of weights and biases are changed in each training step but it also means granularity of the chang. Larger value means faster convergence but it might not reach the exact values(or aprrox.).
    batch_size = 100  # Number of images picked by training algorithms in eac training step.
    training_steps = 2000  # Number of times training algorithm executed with #batch_size images in each iteration.
### [Optimizers available in Tensorflow](https://www.tensorflow.org/api_docs/python/tf/train):
    
    AdadeltaOptimizer
    AdagradDAOptimizer
    AdagradOptimizer
    AdamOptimizer
    FtrlOptimizer
    GradientDescentOptimizer
    MomentumOptimizer
    Optimizer
    ProximalAdagradOptimizer
    ProximalGradientDescentOptimizer
    RMSPropOptimizer
    SyncReplicasOptimizer
    

    


