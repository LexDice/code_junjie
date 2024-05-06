import pyvista as pv
import numpy as np

# file_path = '/Users/junjiezhao/unc/Classes/COMP790/code/new_code/output_testctrl/DeterministicAtlas__EstimatedParameters__ControlPoints.txt'
# with open(file_path) as f:
#     lines = f.readlines()
#
# for one_line in lines:
#     coords = one_line.strip().split(' ')
#     coords = [float(i) for i in coords]
#     print('aa')
#
# aa= float(lines[0])
# print('aa')

mean_obj_path = '/Users/junjiezhao/unc/Classes/COMP790/code/mean_obj/my_mean_obj.vtk'
mesh = pv.read(mean_obj_path)
vertices = np.array(mesh.points).astype(np.float32)
faces = mesh.faces.reshape(-1, 4)
# remove padding of pv polygons(first column)
faces = np.array(faces[:, 1:]).astype(np.int32)

max_val = vertices.max(axis=0)
max_val = np.ceil(max_val)
min_val = vertices.min(axis=0)
min_val = np.floor(min_val)

interval = (max_val - min_val) / [6., 12., 12.]

# generate grid
x = np.linspace(min_val[0], max_val[0], num=6)
y = np.linspace(min_val[1], max_val[1], num=12)
z = np.linspace(min_val[2], max_val[2], num=12)

meshgrid = np.zeros((6*12*12, 3))
l = 0
for i in range(len(x)):
    for j in range(len(y)):
        for k in range(len(z)):
            meshgrid[l, :] = [x[i], y[j], z[k]]
            l += 1

np.savetxt('./test.txt', meshgrid, fmt='%.6f')
print('aa')



