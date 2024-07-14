#DDE_BACKEND=tensorflow.compat.v1 python3 prova_coppelia_external_api.py

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
lx = 1
spheres.append(sim.getObject("/Sphere"))
initial_pos.append(sim.getObjectPosition(spheres[0]))
j = sim.getObjectPosition(spheres[0])[1]
k = 1.0
# create the other spheres
x = np.linspace(0.0, lx, 10)

pinns_plucked.load_model("./pinns_string_force/model/pinns_string_force-20000.ckpt")

for i in x:
    if i == 0.0:
        continue
    s = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [0.1, 0.1, 0.1]) #default radius 0.5
    sim.setObjectColor(s, 0, sim.colorcomponent_ambient_diffuse, [50,1,50])
    sim.setObjectPosition(s, [i, j, k])
    sim.setObjectInt32Parameter(s,sim.shapeintparam_static,0)
    sim.setObjectInt32Parameter(s,sim.shapeintparam_respondable,1)
    spheres.append(copy.deepcopy(s))
    initial_pos.append([i, j, k])

sim.startSimulation()

def func():
    if (t := sim.getSimulationTime()) < 10:
        #print(f'Simulation time: {t:.2f} [s]')
        sim.step()
        pred_pos = pinns_plucked.predict(x, [t])
        #print(pred_pos)
        for i in range(len(spheres)):
            new_pos = [initial_pos[i][0], initial_pos[i][1], 20*pred_pos[i]+initial_pos[i][2]]
            sim.setObjectPosition(spheres[i], new_pos)
        threading.Timer(dt, func).start()
        
    else:
        sim.stopSimulation()
        while (sim.getSimulationState() != sim.simulation_stopped):
            continue
        sim.closeScene()

func()