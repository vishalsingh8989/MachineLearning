"""handwriten_digits.py: Basic nueral networks for identifying handwriiten digits using tensorFlow
Activation: relu and softmax with dropout
Accuracy acheived  :  97.24 %. training_steps = 100000

Test result : 
 ____________________________________________________________________________________________________________________________________________________
|     Activation function             |  Number of layer   | Learning fxn |  Learning_rate  |  Batch_size |  Training_steps | Accuracy |   Total loss | 
|  sigmoid and softmax               |  4                 | GDO          |  0.0005         |   100       |    100000       |  96.57   |    1559.6    | 
|  sigmoid and softmax               |  4                 | GDO          |  0.0002         |   100       |    200000       |          |


"""


__author__ = "Vishal Jasrotia"
__date__   = "Feb, 06, 2018"
__organization__ = "Stony Brook Univerity, NY, USA"
__email__ = "jasrotia.vishal@stonybrook.edu"
__version__ = "0.1"
__status__ = "library_file"


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

