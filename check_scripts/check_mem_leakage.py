
import sys
import gc
import copy
import numpy as np
from memory_profiler import profile

def assign_dict_1(old_dict):
    new_dict = {'1': np.random.rand(512,512,10)}
    array = new_dict['1'][0,0,1]
    old_dict['dict1'] = array

def assign_dict_2(old_dict):
    new_dict = {'1': np.random.rand(512,512,20)}
    old_dict['dict2'] = new_dict

def assign_dict_3(old_dict):
    new_dict = {'1': np.random.rand(512,512,30)}
    old_dict['dict3'] = new_dict

@profile
def main():
    old_dict = {}
    assign_dict_1(old_dict)
    assign_dict_2(old_dict)
    assign_dict_3(old_dict)
    old_dict = {}
    

if __name__ == "__main__":
    main()


