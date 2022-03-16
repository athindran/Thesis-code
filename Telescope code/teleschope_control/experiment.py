"""

This file demonstrates how to use the collisions simulation code.

"""

import numpy as np
import sys
from multiprocessing import Pool

from ics.cobraOps import plotUtils
from ics.cobraOps import targetUtils

from ics.cobraOps.Bench import Bench
from ics.cobraOps.CobrasCalibrationProduct import CobrasCalibrationProduct
from ics.cobraOps.CollisionSimulator import CollisionSimulator
from ics.cobraOps.DistanceTargetSelector import DistanceTargetSelector
from ics.cobraOps.RandomTargetSelector import RandomTargetSelector

def experiment(assiMode=2, rmTEA=True, refElb=True, zeroColl=True):
    # Define the target density to use
    targetDensity = 1.5

    # Load the cobras calibration product
    calibrationProduct = CobrasCalibrationProduct("updatedMaps6.xml")

    # Create the bench instance
    bench = Bench(layout="full", calibrationProduct=calibrationProduct)

    # Generate the targets
    targets = targetUtils.generateRandomTargets(targetDensity, bench)
    nTar = targets.nTargets

    # Select the targets
    selector = DistanceTargetSelector(bench, targets)
    useNegativePhi = selector.run(assiMode=assiMode, rmTEA=rmTEA, refElb=refElb)
    selectedTargets = selector.getSelectedTargets()
    
    # set the Phi directions of the cobras
    bench.cobras.set_useNegativePhi(useNegativePhi)
    
    # Simulate an observation
    simulator = CollisionSimulator(bench, selectedTargets)
    
    if zeroColl:
        # ensure zero collision by naive removal at the final stage
        simulator.zeroCollRun()
    else:
        # no naive removal
        simulator.run()
    
    nCompTar = sum(simulator.assignedCobras)
    nEndColl = simulator.nEndPointCollisions
    nNonEndColl = simulator.nCollisions - simulator.nEndPointCollisions

    return nTar, nCompTar, nEndColl, nNonEndColl

def multi_experi(j, assiMode, rmTEA, refElb, zeroColl):
    np.random.seed(j)
    rec_i = experiment(assiMode=assiMode, rmTEA=rmTEA, refElb=refElb, zeroColl=zeroColl)
    return list(rec_i)

### experiment 1: compare assiMode and rmTEA ###

assiMode = int(sys.argv[1])
rmTEA = sys.argv[2] == 'True'
refElb = sys.argv[3] == 'True'
zeroColl = sys.argv[4] == 'True'
L = 1000000
nThr = 28
rec = []
fout = open('data/endColl_{}_{}_{}_{}.txt'.format(assiMode, rmTEA, refElb, zeroColl), 'w')
for ith in range( int(L/(nThr+1e-6))+1 ):
    with Pool(processes = nThr) as pool:
        jlist = list(range(ith*nThr, min(L, (ith+1)*nThr) ))
        alist = [assiMode]*len(jlist)
        rlist = [rmTEA]*len(jlist)
        slist = [refElb]*len(jlist)
        zlist = [zeroColl]*len(jlist)
        rec_i = pool.starmap(multi_experi, zip(jlist, alist, rlist, slist, zlist))

        # check end collision:
        colli = np.nonzero( np.array(rec_i)[:,2] )[0]
        for idx in colli:
            #print('End Coll at seed: {}'.format(idx + ith*nThr) )
            fout.write(str(idx + ith*nThr))
            fout.flush()
    rec = rec + rec_i
    if (ith+1)%10 == 0:
        print('{} / {}'.format(ith+1, int(L/(nThr+1e-6))+1) )
rec = np.array(rec)
#np.savetxt('data/rec_{}_{}_{}.txt'.format(assiMode, rmTEA, refElb),rec, delimiter = '\t')
np.save('data/rec_{}_{}_{}_{}.npy'.format(assiMode, rmTEA, refElb, zeroColl), rec)
print("assiMode: {}, rmTEA: {}, refElb: {}, zeroColl: {}".format(\
       assiMode, rmTEA, refElb, zeroColl)  )
print("Number of targets: {} +- {}".format(rec[:,0].mean(), np.sqrt(rec[:,0].var())) )
print("Number of completed targets: {} +- {}".format(rec[:,1].mean(), np.sqrt(rec[:,1].var())) )
print("Number of end collisions: {} +- {}".format(rec[:,2].mean(), np.sqrt(rec[:,2].var())) )
print("Number of non-end collisions: {} +- {}".format(rec[:,3].mean(), np.sqrt(rec[:,3].var())) )

############   depricated   ###############
### experiment 2: compare eps ###
'''
fout = open('data/eps_auc.txt', 'w')
fout.write('eps\tnCompTar.m\tnCompTar.std\tnEndColl.m\tnEndColl.std\tnNonEndColl.m\tnNonEndColl.std\n')
eps = np.linspace(0,4,num=101)

for i in range(eps.shape[0]):
    L = 1000
    nThr = 6
    rec = []
    for ith in range( int(L/(nThr+1e-6))+1 ):
        with Pool(processes = nThr) as pool:
            jlist = list(range(ith*nThr, min(L, (ith+1)*nThr) ))
            elist = [eps[i]]*len(jlist)
            rec_i = pool.starmap(multi_experi, zip(jlist, elist))
        rec = rec + rec_i
    rec = np.array(rec)
#    np.savetxt('data/rec.txt',rec[:,2:], delimiter = '\t')
    strr = str(eps[i])
    strr = strr + '\t' + str(rec[:,2].mean()) + '\t' + str(np.sqrt(rec[:,2].var()))
    strr = strr + '\t' + str(rec[:,3].mean()) + '\t' + str(np.sqrt(rec[:,3].var()))
    strr = strr + '\t' + str(rec[:,4].mean()) + '\t' + str(np.sqrt(rec[:,4].var())) + '\n'
    fout.write(strr)
    fout.flush()
    print(i,eps[i])
'''
