import numpy as np
import matplotlib.pyplot as plt

grid_size = 30
x = np.arange(grid_size)
y = np.arange(grid_size)

# hdpoints = np.random.randint(0, 40, size=(20, 6))
hdpoints = np.full(shape=(20,10), fill_value=np.random.randint(0, 40))
ldpoints = np.random.randint(0, 6, size=(20, 2))
# ldpoints = np.array([[19,19], [0,1], [2,1], [1,2] ,[0,3 ],
#                      [0,18], [2,19], [10, 19], [4,17], [3,20],
#                     [19,5], [18,10], [10,10], [10, 20], [10,12],
#                      [19,17,], [16,15], [14, 0], [12, 5], [16,4]])

hdpoint = np.random.randint(80, 100, size=(1, 10))
xv, yx = np.meshgrid(x, y)

hd_dist= np.linalg.norm(hdpoints- hdpoint, axis=1)

ld_dist = np.zeros((grid_size,grid_size, ldpoints.shape[0]))

for i in range(20):
    xy =  np.dstack((xv - ldpoints[i, 0], yx - ldpoints[i, 1]))
    distance_one_p = np.linalg.norm(xy, axis=2)
    ld_dist[... ,i] = distance_one_p

diffs = np.abs(ld_dist - hd_dist)
error = np.sum((diffs/hd_dist), axis=2)




h = plt.contourf(x, y, error)
plt.axis('scaled')
plt.colorbar()
plt.show()