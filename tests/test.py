from hdimvis.data_fetchers.DataFetcher import DataFetcher
import numpy as np
import matplotlib.pyplot as plt




latitude = np.pi * np.linspace(-1 / 2, 1 / 2, 20)
longitude = np.pi * np.linspace(0, 2, 20)

z = np.sin(latitude)

x = np.cos(latitude)[:, None] * np.sin(longitude)[None, :]
y = np.cos(latitude)[:, None] * np.cos(longitude)[None, :]
z = np.tile(z[:,None], (1,20))



# ax = fig.add_subplot()



xm = np.tile(latitude[:,None], (1,20)) * 3
ym = np.tile(longitude[None,:], (20,1)) * 3

# ax.scatter(xm, ym)

meridian_zero = (ym.max() + ym.min())/2
equator = (xm.max() + xm.min())/2
radius = 3


pitch = (ym - equator / radius)
yaw = (xm) / radius

z1 = np.sin(pitch)
x1 = np.cos(pitch) * np.sin(yaw)
y1= np.cos(pitch) * np.cos(yaw)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, y1, z1 )
plt.show()
