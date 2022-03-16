# Implementation of Target Assignment 
This branch contains implementation of target-cobra assignment problem with end-collision avoidance.
- update on 05/10/2021: Merge transient collision avoidance algorithms from branch OnlineCollisionSimulator.

The code is adapted from [Subaru-PFS](https://github.com/Subaru-PFS/ics_cobraOps).

We adopt a sparse version of Auction algorithm to help maximize the number of target completion.

The end collisions are classified and solved by the following techniques.
1. target-target collision -- avoided by safe selection and minimizing target's local density
2. target-elbow-arm collision -- avoided by elbow reflection and naive abandonment
3. target-unassigned-cobra collision -- avoided by maximizing center-line-to-target distances

# code run
demo.py -- analyze and plot the performance in a singal trial.

demo_athindran.py -- test transient collision avoidance with Athindran's code.

experiment.py -- analyze multiple trials, speeded up by multiprocessing.


# Major changes in ./ics/cobraOps

## AuctionTargetSelector.py
run(): a. add AssiMode, rmTEC, and refElb features.
       b. return useNegativePhi for elbow reflection.

run() chooses the assignment algorithm based on AssiMode:
- AssiMode >=2: Auction algorithm with safe assignment and target local density minimization
- AssiMode ==1: Auction algorithm with safe assignment
- AssiMode ==0: Greedy algorithm with safe assignment
- AssiMode ==-1: Auction algorithm
- Else: Greedy algorithm

detectTarElbowArmCollisions(): detect target-elbow-arm collisions under a certain configuration of useNegativePhi.

selectTargetsBySafeTargets(): solve the assignment problem by Greedy algorithm with safe selection

selectTargetsBySafeAuction(): solve the assignment problem by Auction algorithm with safe selection

selectTargetsByWeightedAuction(): solve the assignment problem by Auction algorithm with safe selection and target local density minimization

## CollisionSimulator.py
zeroCollRun(): ensure zero collision by naive removal at the final stage

optimizeUnassignedCobraPositions2(): set the location of unassigned cobras by maximizing center-line-to-target distances

distancesToLineSegments(): compute point-to-line-segment distances. This is a slight modification from the function in Bench.py

## CobraGroup.py
set_useNegativePhi(): register self.useNegativePhi

calculateElbowPositions(): calculate all elbow positions under self.useNegativePhi

calculateRotationAngles(): calculate all rotation angles under self.useNegativePhi

## TargetSelector.py
calculateAccessibleTargets(): compute accessibleTargetElbows of the cobra in both positive and negative phi

