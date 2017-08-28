import numpy as np
import random

def createMap(size=5):
    map = np.array([[0 for col in range(size)] for rows in range(size)])

    # placing start value
    start = ( int(size/2) , int(size/2) )
    map[start] = 1

    # placing obstacles
    map[int(size/2)+1:, 0:int(size/5)] = -1000
    map[int(size/size):int(size/3)+1, :int(size/2)] = -1000
    map[int(size/2)+1:, int((size-1)/2):int(size/2)+1] = -1000
    map[int(size/size):int(size/2)+1, size-int(size/5):] = -1000
    return map, start

def goLowest():
    pass

def goAnyZero():
    pass

def goToStart():
    pass


map, start = createMap(size=5)



