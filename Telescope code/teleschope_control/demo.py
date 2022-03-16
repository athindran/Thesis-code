"""

This file demonstrates how to use the collisions simulation code.

"""

import numpy as np
import time as time

from ics.cobraOps import plotUtils
from ics.cobraOps import targetUtils

from ics.cobraOps.Bench import Bench
from ics.cobraOps.CobrasCalibrationProduct import CobrasCalibrationProduct
from ics.cobraOps.CollisionSimulator import CollisionSimulator
from ics.cobraOps.DistanceTargetSelector import DistanceTargetSelector
from ics.cobraOps.PriorityTargetSelector import PriorityTargetSelector
from ics.cobraOps.AuctionTargetSelector import AuctionTargetSelector

np.random.seed(10) #146236, 505432, 4768

# Define the target density to use
targetDensity = 10

# Define the selector mode (Distance, Priority, Auction)
selector_mode = 'Auction'
#selector_mode = 'Priority'

# Load the cobras calibration product
calibrationProduct = CobrasCalibrationProduct("updatedMaps6.xml")

# Create the bench instance
bench = Bench(layout="full", calibrationProduct=calibrationProduct)
print("Number of cobras:", bench.cobras.nCobras)

# Generate the targets
targets = targetUtils.generateRandomTargets(targetDensity, bench)
targets.priorities = np.random.uniform(size = targets.nTargets)
print("Number of simulated targets:", targets.nTargets)


# Select the targets
start = time.time()
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
simulator = CollisionSimulator(bench, selectedTargets)

# ensure zero collision by naive removal at the final stage
#simulator.zeroCollRun()

# no naive removal
simulator.run()

nCompTar = sum(simulator.assignedCobras)
totalPriority = sum(simulator.targets.priorities[simulator.assignedCobras])
print("Total Priority: ", totalPriority)
print("Number of completed targets: ", nCompTar)
print("Number of cobras involved in collisions:", simulator.nCollisions)
print("Number of cobras unaffected by end collisions: ", simulator.nCollisions - simulator.nEndPointCollisions)
print("Total simulation time (s):", time.time() - start)

# Plot the simulation results
## plot selected targets
simulator.plotResults(extraTargets=selectedTargets, paintFootprints=False)

## plot all targets
#simulator.plotResults(extraTargets=targets, paintFootprints=False)

# Animate one of the trajectory collisions
(problematicCobras,) = np.where(np.logical_and(simulator.collisions, simulator.endPointCollisions == False))

if len(problematicCobras) > 0:
    simulator.animateCobraTrajectory(problematicCobras[0], extraTargets=targets)

# Pause the execution to have time to inspect the figures
plotUtils.pauseExecution()
