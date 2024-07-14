#DDE_BACKEND=tensorflow.compat.v1 python3 Membrane_with_Force.py


from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import copy
import threading
import os

import pinns_membrane_force

client = RemoteAPIClient()
sim = client.require('sim')

sim.setStepping(True)

dt = 0.01

sim.loadScene(os.path.abspath("./Initial_scene.ttt"))
sim.setFloatParam(sim.floatparam_simulation_time_step , dt)


spheres = []
initial_pos = []
lx = 1
ly = 1
radius = 0.1
num_spheres = 20

spheres.append(sim.getObject("/Sphere"))
initial_pos.append(sim.getObjectPosition(spheres[0]))
j = sim.getObjectPosition(spheres[0])[1]
k = 1.0
# create the other spheres
x = np.linspace(0.0, lx, 20)
y = np.linspace(0.0, ly, 20)

pinns_membrane_force.load_model("./pinns_membrane_force/model/model-20000.ckpt")

for j in y:
    for i in x:
        if (i == 0.0) and (j == 0.0):
            continue
        s = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [radius, radius, radius])
        sim.setObjectPosition(s, [i, j, k])
        sim.setObjectColor(s, 0, sim.colorcomponent_ambient_diffuse, [50,1,50])
        sim.setObjectInt32Parameter(s, sim.shapeintparam_static,1)
        sim.setObjectInt32Parameter(s, sim.shapeintparam_respondable,1)
        spheres.append(copy.deepcopy(s))
        initial_pos.append([i, j, k])

sim.startSimulation()

def func():
    if (t := sim.getSimulationTime()) < 10:
        #print(f'Simulation time: {t:.2f} [s]')
        sim.step()
        pred_pos = pinns_membrane_force.predict(x, y, [t])
        #print(pred_pos)
        for j in range(len(y)):
            for i in range(len(x)):
                new_pos = [initial_pos[(num_spheres*i+j)][0], initial_pos[(num_spheres*i+j)][1], -20*pred_pos[i,j]+initial_pos[(num_spheres*i+j)][2]]
                sim.setObjectPosition(spheres[(num_spheres*i+j)], new_pos)
        threading.Timer(dt, func).start()
        
    else:
        sim.stopSimulation()
        while (sim.getSimulationState() != sim.simulation_stopped):
            continue
        sim.closeScene()

func()