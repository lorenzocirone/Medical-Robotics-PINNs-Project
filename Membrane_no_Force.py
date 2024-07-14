#DDE_BACKEND=tensorflow.compat.v1 python3 Membrane_no_Force.py

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import copy
import threading
import os

import pinns_plucked

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

dt = 0.01

sim.loadScene(os.path.abspath("./Initial_scene.ttt"))
sim.setFloatParam(sim.floatparam_simulation_time_step , dt)


spheres = []
initial_pos = []
radius = 0.1
num_spheres = 20

lx = 1
ly = 1
spheres.append(sim.getObject("/Sphere"))
initial_pos.append(sim.getObjectPosition(spheres[0]))


j = sim.getObjectPosition(spheres[0])[1]

k = 1.0
# create the other spheres
x = np.linspace(0.0, lx, num_spheres)
y = np.linspace(0.0, ly, num_spheres)
xx, yy = np.meshgrid(x, y)  # Create meshgrid for 2D positions
print(os. getcwd())
# ckpt_path ="./sig50_C10/sig50_C10-40000.ckpt" 
pinns_plucked.load_model("sig50_C10-40000.ckpt")
#for i in x:
#    if i == 0.0:
#        continue
#    s = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [0.05, 0.05, 0.05])
#    sim.setObjectPosition(s, [i, j, k])
#    sim.setObjectInt32Parameter(s,sim.shapeintparam_static,0)
#    sim.setObjectInt32Parameter(s,sim.shapeintparam_respondable,1)
#    spheres.append(copy.deepcopy(s))
#    initial_pos.append([i, j, k])
#
#    s2 = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [0.05, 0.05, 0.05])
#    sim.setObjectPosition(s2, [i, j2, k])
#    sim.setObjectInt32Parameter(s2,sim.shapeintparam_static,0)
#    sim.setObjectInt32Parameter(s2,sim.shapeintparam_respondable,1)
#    spheres2.append(copy.deepcopy(s))
#    initial_pos2.append([i, j2, k])
#
#sim.startSimulation()

for i in range(len(x)):
    for j in range(len(y)):
        if i == 0 and j == 0:  # Skip the first element (already a sphere there)
            continue
        s = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [radius, radius, radius])
        sim.setObjectColor(s, 0, sim.colorcomponent_ambient_diffuse, [50,1,50])
        pos = [xx[i, j], yy[i, j], k]  # Initial position with zero z-coordinate
        sim.setObjectPosition(s, pos)
        sim.setObjectInt32Parameter(s,sim.shapeintparam_static,1)
        sim.setObjectInt32Parameter(s,sim.shapeintparam_respondable,1)
        spheres.append(s)
        initial_pos.append(pos)

sim.startSimulation()

# def func():
#     if (t := sim.getSimulationTime()) < 10:
#         #print(f'Simulation time: {t:.2f} [s]')
#         sim.step()
#         pred_pos = pinns_plucked.predict(x, [t])
#         
#         #print(pred_pos)
#         for i in range(len(spheres)):
#             new_pos = [initial_pos[i][0], initial_pos[i][1], pred_pos[i]+initial_pos[i][2]]
#             sim.setObjectPosition(spheres[i], new_pos)
# 
#         threading.Timer(dt, func).start()
#         
#     else:
#         sim.stopSimulation()
#         while (sim.getSimulationState() != sim.simulation_stopped):
#             continue
#         sim.closeScene()
# 
# func()

def func():
    if (t := sim.getSimulationTime()) < 10:
            #print(f'Simulation time: {t:.2f} [s]')
            sim.step()
            pred_pos = pinns_plucked.predict(x, [t])
            #print(len(spheres))
            j = 0
            for i in range(len(spheres)):
                if j % num_spheres == 0:
                    j = 0
                new_pos = [initial_pos[i][0], initial_pos[i][1], pred_pos[j]+initial_pos[i][2]]
                sim.setObjectPosition(spheres[i], new_pos)
                j += 1
            threading.Timer(dt, func).start()

    else:
        sim.stopSimulation()
        while (sim.getSimulationState() != sim.simulation_stopped):
            continue
        sim.closeScene()
 
func()