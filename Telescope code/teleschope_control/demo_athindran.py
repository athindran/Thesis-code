"""

This file demonstrates how to use the collisions simulation code.

"""

import numpy as np
import time as time
import sys
sys.path.append("/home/athindran/cobrafibrecontrol/teleschope_control/ics")
from ics.cobraOps import plotUtils
from ics.cobraOps import targetUtils

from ics.cobraOps.Bench import Bench
from ics.cobraOps.CobrasCalibrationProduct import CobrasCalibrationProduct
from ics.cobraOps.CollisionSimulator import CollisionSimulator
#from ics.cobraOps.OnlineCollisionSimulatorEL import OnlineCollisionSimulator
#from ics.cobraOps.ConflictBasedSearch import ConflictBasedSearch
#from ics.cobraOps.ConflictBasedSearchEL import ConflictBasedSearch

# Parallelized version
#from ics.cobraOps.ConflictBasedSearch_pp import ConflictBasedSearch
from ics.cobraOps.ConflictBasedSearchEL_pp import ConflictBasedSearch

from ics.cobraOps.DistanceTargetSelector import DistanceTargetSelector
from ics.cobraOps.PriorityTargetSelector import PriorityTargetSelector
from ics.cobraOps.AuctionTargetSelector import AuctionTargetSelector

np.random.seed(None) #146236, 505432

a_list = []
b_list = []
c_list = []
d_list = []

e_list = []
f_list = []
g_list = []
h_list = []

i_list = []
j_list = []
k_list = []

subgraphsizes = []
collseeds = []
unaccseeds = []
for see in range(77,78):
    np.random.seed(see)
    print("seed:",see)
    #np.random.seed(5)


    # Define the target density to use
    targetDensity = 10

    # Define the selector mode (Distance, Priority, Auction)
    #selector_mode = 'Auction'
    selector_mode = 'Priority'

    # Load the cobras calibration product
    calibrationProduct = CobrasCalibrationProduct("updatedMaps6.xml")

    # Create the bench instance
    bench = Bench(layout="full", calibrationProduct=calibrationProduct)
    print("Number of cobras:", bench.cobras.nCobras)

    # Generate the targets and random priorities
    targets = targetUtils.generateRandomTargets(targetDensity, bench)
    targets.priorities = np.random.uniform(size = targets.nTargets)
    print("Number of simulated targets:", targets.nTargets)

    # Select the targets
    if selector_mode == 'Auction':
        selector = AuctionTargetSelector(bench, targets)
        useNegativePhi = selector.run(assiMode=1, rmTEA=True, refElb=True)
        # set the Phi directions of the cobras
        bench.cobras.set_useNegativePhi(useNegativePhi)
    else:
        if selector_mode == 'Priority':
            selector = PriorityTargetSelector(bench, targets)
        else:
            selector = DistanceTargetSelector(bench, targets)
        selector.run()
    selectedTargets = selector.getSelectedTargets()

    # Simulate an observation
    start = time.time()
    
    # original simulator
    #simulator = CollisionSimulator(bench, selectedTargets, 340)
    #simulator = CollisionSimulator(bench, selectedTargets)
    
    # To use the Lookahead collision avoidance, use the OnlineCollisionSimulator
    #simulator = OnlineCollisionSimulator(bench, selectedTargets, avertCollisions=True)
    
    # To use the Conflict based search, use this
    simulator = ConflictBasedSearch(bench, selectedTargets, 380)
    #simulator = ConflictBasedSearch(bench, selectedTargets)
    #simulator = ConflictBasedSearch(bench, selectedTargets)
    
    # ensure zero collision by naive removal at the final stage
    #simulator.zeroCollRun()

    # no naive removal
    simulator.run(solveCollisions=True)
    
    nCompTar = sum(simulator.assignedCobras)
    totalPriority = sum(simulator.targets.priorities[simulator.assignedCobras])
    print("Total Priority: ", totalPriority)
    print("Number of completed targets: ", nCompTar)
    print("Number of cobras involved in collisions:", simulator.nCollisions)
    print("Number of cobras involved in end collisions:", simulator.nEndPointCollisions)
    print("Number of cobras involved in non-end collisions: ", simulator.nCollisions - simulator.nEndPointCollisions)
    a_list.append(simulator.nCollisions)
    b_list.append(simulator.nEndPointCollisions)
    c_list.append(simulator.nCollisions - simulator.nEndPointCollisions)
    d_list.append(time.time() - start)
    len_list = simulator.lenlist()
    [num_acq,unacquiredcobras] = simulator.numacq()
    #mulcol = simulator.multiplecollisions
    if simulator.nCollisions>0:
        collseeds.append(see)
    if len(unacquiredcobras)>0:
        print("Unacquired:",unacquiredcobras)
        #print(simulator.trajectories.tht[unacquiredcobras[0],:])
        #print(simulator.trajectories.finalTht[unacquiredcobras[0]])
        unaccseeds.append(see)
    #print(simulator.trajectories.tht[1887,:])
    #print(simulator.trajectories.finalTht[1887])     
    #for l in range(len(simulator.cobraSubgraphs)):
    #  subgraphsizes.append(len(simulator.cobraSubgraphs[l]))
    #print(unacquiredcobras)
    #print(simulator.trajectories.tht[unacquiredcobras,:])
    #print(simulator.trajectories.finalTht[unacquiredcobras])

    #if(simulator.nCollisions>0):
    #    print("Stop:")
    #    break;
    #if np.size(unacquiredcobras)>0:
    #    print("Unacquired:",i)
    #    break;
    num_stops = simulator.numstops()
    print('max:',np.max(len_list))
    print('min:',np.min(len_list))
    print('mean:',np.mean(len_list))
    print('std:',np.std(len_list))
    
    #print("Where:",np.where(np.array(len_list)==0))
    #print("check1:",simulator.trajectories.tht[40,:])
    #print("check2:",simulator.trajectories.phi[40,:])
    #print("Check1:",simulator.trajectories.finalTht[40])
    #print("Check2:",simulator.trajectories.finalPhi[40])
    
    e_list.append(np.max(len_list))
    f_list.append(np.min(len_list))
    g_list.append(np.mean(len_list))
    h_list.append(np.std(len_list))

    i_list.append(num_acq)
    j_list.append(num_stops)
    print('num_acq:',num_acq)


    print("Total simulation time (s):", time.time() - start)
    k_list.append(time.time() - start)

    # Plot the simulation results
    #simulator.plotResults(extraTargets=targets, paintFootprints=False)

print("Number of cobras involved in collisions:",np.mean(a_list),'+-',np.std(a_list))
print("Number of cobras involved in end collisions:", np.mean(b_list),'+-',np.std(b_list))
print("Number of cobras involved in non-end collisions: ", np.mean(c_list),'+-',np.std(c_list))
print("Total simulation time (s): ", np.mean(d_list),'+-',np.std(d_list))
# FFT reduction of targets
#targets.FFT_reduction( int(bench.cobras.nCobras*(1+ (targetDensity-1)*0.0)) )
#print(simulator.trajectories.tht[221,:])
#print(simulator.trajectories.tht[80,:])
#print(simulator.trajectories.phi[221,:])
#print(simulator.trajectories.phi[80,:])


print("mean of max length (s): ", np.mean(e_list),'+-',np.std(e_list))
print("mean of min length (s): ", np.mean(f_list),'+-',np.std(f_list))
print("mean of mean length (s): ", np.mean(g_list),'+-',np.std(g_list))

#print("mean subgraph size: ",np.mean(subgraphsizes),'+-',np.std(subgraphsizes))
#print("Max subgraph size: ", np.max(subgraphsizes))
#print("Min subgraph size: ", np.min(subgraphsizes))

# Plot the simulation results
## plot selected targets
#simulator.plotResults(extraTargets=selectedTargets, paintFootprints=False)

## plot all targets
#simulator.plotResults(extraTargets=targets, paintFootprints=False)

print("mean of acq (s): ", np.mean(i_list),'+-',np.std(i_list))
print("mean no. of stops (s): ", np.mean(j_list),'+-',np.std(j_list))
print("Collision seeds: ",collseeds)
print("Unacquired seeds: ",unaccseeds)
print("Mean simulation time :", np.mean(k_list),'+-',np.std(k_list))

# Animate one of the trajectory collisions
(problematicCobras,) = np.where(np.logical_and(simulator.collisions, simulator.endPointCollisions == False))
print("Colliding cobras:",problematicCobras)
#if len(problematicCobras) > 0:
# simulator.animateCobraTrajectory(problematicCobras[0], extraTargets=targets, fileName = "test42cobras.mp4")
#simulator.animateCobraTrajectory(1587, extraTargets=targets, fileName = "testcbs3.mp4")

#simulator.animateCobraTrajectory(19, extraTargets=targets, fileName = "CBS_test2.mp4")
# Pause the execution to have time to inspect the figures
#plotUtils.pauseExecution()
#array([ 809,  811,  838,  839,  990, 1018, 1078, 1097, 1098, 2286, 2288],
#      dtype=int64)

#mean subgraph size:  13.316488222698073 +- 6.9341075887859684
#Max subgraph size:  69
#Min subgraph size:  6
