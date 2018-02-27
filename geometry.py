import numpy as np

p = np.array((0, .25))
x = np.arange(-.002, .0021, .0001)
xy = np.pad(np.reshape(x,(-1,1)),((0,0),(0,1)),'constant')

vec = p-xy
vec = np.divide(vec,np.reshape(np.linalg.norm(vec,axis=1),(-1,1)))
