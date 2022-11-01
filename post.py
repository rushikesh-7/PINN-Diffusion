import numpy as np
import matplotlib.pyplot as plt 

d=[]
p=[]
# read=[12,13,14]
read=[1]
for i in read:
    name = 'data_' + str(i) + '.dat'
    arr = np.loadtxt(name)
    d.append(arr)
    name2 = 'points_' + str(i) + '.dat'
    arr2 = np.loadtxt(name2)
    p.append(arr2)

T = d[0][:,0]
X = d[0][:,1]

sol1 = d[0][:,2]
# sol2 = d[1][:,2] 
# sol3 = d[2][:,2]

sol = d[0][:,3]

t_0 = p[0][:100, 0]
t_b = p[0][101:200, 0]
t_r = p[0][201:2200, 0]
# t_e_9 = p[0][2200:, 0]
# t_e_10 = p[1][2200:, 0]
# t_e_11 = p[2][2200:, 0]

x_0 = p[0][:100, 1]
x_b = p[0][101:200, 1]
x_r = p[0][201:2200, 1]
# x_e_9 = p[0][2200:, 1]
# x_e_10 = p[1][2200:, 1]
# x_e_11 = p[2][2200:, 1]

climlow = np.abs(np.min(sol1))
# climlow2 = np.abs(np.min(sol-sol2))
# climlow3 = np.abs(np.min(sol-sol3))
# climlow = min(climlow1, climlow2, climlow3)
climmax = np.abs(np.max(sol1))
# climmax2 = np.abs(np.max(sol-sol2))
# climmax3 = np.abs(np.max(sol-sol3))
# climmax = max(climmax1, climmax2, climmax3)

lim = max(climlow, climmax)

fig = plt.figure(figsize=(16,9))

# plt.subplot(3,1,1)
plt.scatter(T,X, s=1,c=sol1, cmap='seismic')
plt.colorbar()
plt.clim(-lim, lim)
plt.scatter(t_0, x_0, c='c', marker='x', s=20)
plt.scatter(t_b, x_b, c='g', marker='x', s=20)
plt.scatter(t_r, x_r, c='m', marker='.', s=1)
# plt.scatter(t_e_9, x_e_9, c='black', marker='o', s=10)
plt.xlabel('t')
plt.ylabel('x')

# plt.subplot(3,1,2)
# plt.scatter(T,X, s=1,c=sol-sol2, cmap='seismic')
# plt.colorbar()
# plt.clim(-lim, lim)
# plt.scatter(t_0, x_0, c='c', marker='x', s=20)
# plt.scatter(t_b, x_b, c='g', marker='x', s=20)
# plt.scatter(t_r, x_r, c='m', marker='.', s=1)
# plt.scatter(t_e_10, x_e_10, c='black', marker='o', s=10)
# plt.xlabel('t')
# plt.ylabel('x')

# plt.subplot(3,1,3)
# plt.scatter(T,X, s=1,c=sol-sol3, cmap='seismic')
# plt.colorbar()
# plt.clim(-lim, lim)
# plt.scatter(t_0, x_0, c='c', marker='x', s=20)
# plt.scatter(t_b, x_b, c='g', marker='x', s=20)
# plt.scatter(t_r, x_r, c='m', marker='.', s=1)
# plt.scatter(t_e_11, x_e_11, c='black', marker='o', s=10)
# plt.xlabel('t')
# plt.ylabel('x')

# fig = plt.figure(figsize=(16,9))
# plt.scatter(T, X, s=1, c=sol1, cmap='seismic')
# plt.colorbar()
# plt.xlabel('t')
# plt.ylabel('x')
plt.savefig("PINN_1.png")
# plt.show()
