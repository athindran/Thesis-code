"""

DistanceTargetSelector class.

Consult the following papers for more detailed information:

  https://ui.adsabs.harvard.edu/abs/2012SPIE.8450E..17F
  https://ui.adsabs.harvard.edu/abs/2014SPIE.9151E..1YF
  https://ui.adsabs.harvard.edu/abs/2016arXiv160801075T
  https://ui.adsabs.harvard.edu/abs/2018SPIE10707E..28Y
  https://ui.adsabs.harvard.edu/abs/2018SPIE10702E..1CT

"""

import numpy as np

from .TargetSelector import TargetSelector
from .cobraConstants import NULL_TARGET_INDEX, COBRA_LINK_RADIUS, COBRA_LINK_LENGTH
from queue import Queue
from scipy.spatial import cKDTree

class AuctionTargetSelector(TargetSelector):
    """Subclass of the TargetSelector class used to select optimal targets for
    a given PFI bench. The selection criteria is based on the target-to-cobra
    distance.

    """
    def run(self, maximumDistance=np.Inf, assiMode=2, rmTEA=True, refElb=True, eps=None, FFT=False):
        """Runs the whole target selection process assigning a single target to
        each cobra in the bench.

        Parameters
        ----------
        maximumDistance: float, optional
            The maximum radial distance allowed between the targets and the
            cobra centers. Default is no limit (the maximum radius that the
            cobra can reach).
        solveCollisions: bool, optional
            If True, the selector will try to solve cobra end-point collisions
            assigning them alternative targets. Default is True.

        """
        # Construct a KD tree if the target density is large enough
        if self.targets.nTargets / self.bench.cobras.nCobras > 50:
            self.constructKDTree()

        # Obtain the accessible targets for each cobra ordered by distance
        self.calculateAccessibleTargets(maximumDistance)
        if FFT:
            self.reduceByNonAccAndFFT(eps = eps)
            if self.targets.nTargets / self.bench.cobras.nCobras > 50:
                self.constructKDTree()
            self.calculateAccessibleTargets(maximumDistance)
            self.selectTargetsByAuction()
        else:
            # Select a single target for each cobra
            if assiMode>=2:
                self.selectTargetsByWeightedAuction(assiMode)
            elif assiMode==1:
                self.selectTargetsBySafeAuction()
            elif assiMode==0:
                self.selectSafeTargets()
            elif assiMode==-1:
                self.selectTargetsByAuction()
            else:
                self.selectTargets()
            
        
        if rmTEA or refElb:
            # determine positive or negative orientation of phi motor
            # this symmetry help remove tar-elb-arm collision
            useNegativePhi = [True]*len(self.assignedTargetIndices)
            collidedCob = self.detectTarElbowArmCollisions(useNegativePhi)
            if refElb:
                for cob in collidedCob:
                    useNegativePhi[cob] = not useNegativePhi[cob]
                collidedCob = self.detectTarElbowArmCollisions(useNegativePhi)
            
            # if cannot remove by symmetry, abandon the target
            if rmTEA:
                for cob in collidedCob:
                    self.assignedTargetIndices[cob] = -1   
            return useNegativePhi
        return None

    def selectTargets(self):
        """Selects a single target for each cobra based on their distance to
        the cobra center.

        This method should always be run after the calculateAccessibleTargets
        method.

        """
        # Create the array that will contain the assigned target indices
        self.assignedTargetIndices = np.full(
            self.bench.cobras.nCobras, NULL_TARGET_INDEX)
        
        # Assign targets to cobras, starting from the closest to the more far
        # away ones
        freeCobras = np.full(self.bench.cobras.nCobras, True)
        freeTargets = np.full(self.targets.nTargets, True)
        maxTargetsPerCobra = self.accessibleTargetIndices.shape[1]

        for i in range(maxTargetsPerCobra):
            # Get a list with the unique targets in this column
            indices = self.accessibleTargetIndices[:, i]
            uniqueIndices = np.unique(indices[freeCobras])

            # Remove from the list the NULL target index value if it's present
            uniqueIndices = uniqueIndices[uniqueIndices != NULL_TARGET_INDEX]

            # Select only those targets that are free
            uniqueIndices = uniqueIndices[freeTargets[uniqueIndices]]

            # Loop over the unique target indices
            for targetIndex in uniqueIndices:
                # Get the free cobras for which this target is the closest
                (cobraIndices,) = np.where(
                    np.logical_and(indices == targetIndex, freeCobras))

                # Check how many cobras we have
                if len(cobraIndices) == 1:
                    # Use this single cobra for this target
                    cobraToUse = cobraIndices[0]
                else:
                    # Select the cobras for which this is the only target
                    accessibleTargets = self.accessibleTargetIndices[
                        cobraIndices, i:]
                    targetIsAvailable = accessibleTargets != NULL_TARGET_INDEX
                    targetIsAvailable[targetIsAvailable] = freeTargets[
                        accessibleTargets[targetIsAvailable]]
                    nAvailableTargets = np.sum(targetIsAvailable, axis=1)
                    singleTargetCobras = cobraIndices[nAvailableTargets == 1]

                    # Decide depending on how many of these cobras we have
                    if len(singleTargetCobras) == 0:
                        # All cobras have multiple targets: select the closest
                        distances = self.accessibleTargetDistances[
                            cobraIndices, i]
                        cobraToUse = cobraIndices[distances.argmin()]
                    elif len(singleTargetCobras) == 1:
                        # Assign the target to the only single target cobra
                        cobraToUse = singleTargetCobras[0]
                    else:
                        # Assign the target to the closest single target cobra
                        distances = self.accessibleTargetDistances[
                            singleTargetCobras, i]
                        cobraToUse = singleTargetCobras[distances.argmin()]

                # Assign the target to the selected cobra
                self.assignedTargetIndices[cobraToUse] = targetIndex
                freeCobras[cobraToUse] = False
                freeTargets[targetIndex] = False

    def selectSafeTargets(self):
        # build targets and target search tree
        targets = (np.unique(self.accessibleTargetIndices)[1:]).tolist() # ignore target -1
        poss = self.targets.positions[targets]
        tree = cKDTree(np.column_stack((poss.real, poss.imag)))
        tar2tid = {tar:id for id, tar in enumerate(targets)}
        
        # Create the array that will contain the assigned target indices
        self.assignedTargetIndices = np.full(
            self.bench.cobras.nCobras, NULL_TARGET_INDEX)
        
        # Assign targets to cobras, starting from the closest to the more far
        # away ones
        freeCobras = np.full(self.bench.cobras.nCobras, True)
        freeTargets = np.full(self.targets.nTargets, True)
        maxTargetsPerCobra = self.accessibleTargetIndices.shape[1]

        for i in range(maxTargetsPerCobra):
            # Get a list with the unique targets in this column
            indices = self.accessibleTargetIndices[:, i]
            uniqueIndices = np.unique(indices[freeCobras])

            # Remove from the list the NULL target index value if it's present
            uniqueIndices = uniqueIndices[uniqueIndices != NULL_TARGET_INDEX]

            # Select only those targets that are free
            uniqueIndices = uniqueIndices[freeTargets[uniqueIndices]]

            # Loop over the unique target indices
            for targetIndex in uniqueIndices:
                if not freeTargets[targetIndex]: continue
                # Get the free cobras for which this target is the closest
                (cobraIndices,) = np.where(
                    np.logical_and(indices == targetIndex, freeCobras))

                # Check how many cobras we have
                if len(cobraIndices) == 1:
                    # Use this single cobra for this target
                    cobraToUse = cobraIndices[0]
                else:
                    # Select the cobras for which this is the only target
                    accessibleTargets = self.accessibleTargetIndices[
                        cobraIndices, i:]
                    targetIsAvailable = accessibleTargets != NULL_TARGET_INDEX
                    targetIsAvailable[targetIsAvailable] = freeTargets[
                        accessibleTargets[targetIsAvailable]]
                    nAvailableTargets = np.sum(targetIsAvailable, axis=1)
                    singleTargetCobras = cobraIndices[nAvailableTargets == 1]

                    # Decide depending on how many of these cobras we have
                    if len(singleTargetCobras) == 0:
                        # All cobras have multiple targets: select the closest
                        distances = self.accessibleTargetDistances[
                            cobraIndices, i]
                        cobraToUse = cobraIndices[distances.argmin()]
                    elif len(singleTargetCobras) == 1:
                        # Assign the target to the only single target cobra
                        cobraToUse = singleTargetCobras[0]
                    else:
                        # Assign the target to the closest single target cobra
                        distances = self.accessibleTargetDistances[
                            singleTargetCobras, i]
                        cobraToUse = singleTargetCobras[distances.argmin()]

                # Assign the target to the selected cobra
                self.assignedTargetIndices[cobraToUse] = targetIndex
                freeCobras[cobraToUse] = False
                #freeTargets[targetIndex] = False
                
                # update target availability to prevent end collision
                tg = tree.query_ball_point(tree.data[tar2tid[targetIndex]], (COBRA_LINK_RADIUS*2)+0.01)
                for tid in tg:
                    freeTargets[targets[tid]] = False
                        
    def reduceByNonAccAndFFT(self, eps=None):
        nCobras = self.accessibleTargetIndices.shape[0]
        accessible = np.unique(self.accessibleTargetIndices)[1:].tolist()
        
        self.targets = self.targets.select(accessible) # remove nonAccessible Targets
        if eps is None: eps = COBRA_LINK_RADIUS*2+0.01
 
        self.targets.FFT_reduction(eps = eps)
    
    def detectTarElbowArmCollisions(self, useNegativePhi):
        if self.kdTree is None:
            self.constructKDTree()
        
        assignedTar = set(self.assignedTargetIndices)
        assignedTar.remove(-1)
        collidedTar, collidedCob = set(), set()
        for cob in range(len(self.assignedTargetIndices)):
            tar = self.assignedTargetIndices[cob]
            if tar == -1: continue

            i = np.where( self.accessibleTargetIndices[cob]==tar )[0][0]
            if useNegativePhi[cob]:
                elbPos = self.accessibleTargetElbows[cob,i]
            else:
                elbPos = self.accessibleTargetElbows2[cob,i]
                
            # remove Target-Elbow collision
            _, indices = self.kdTree.query([elbPos.real, elbPos.imag],
                                           k=None,
                                           distance_upper_bound=(COBRA_LINK_RADIUS*2)+0.01)
            for ind in indices:
                if ind in assignedTar:
                    collidedTar.add(ind)
                    collidedCob.add(cob)
            
            # remove Target-Arm collision
            endPos = self.targets.positions[tar]
            midPos = (endPos + elbPos)/2
            _, indices = self.kdTree.query([midPos.real, midPos.imag],
                                           k=None,
                                           distance_upper_bound=np.sqrt(4*(COBRA_LINK_RADIUS**2)+(COBRA_LINK_LENGTH**2)/4)+0.01)
            if len(indices)<=1:
                continue
            indices.remove(tar)
            
            grad = endPos - elbPos
            grad = grad / np.abs(grad)
            vmax = ( np.conj(grad)*endPos ).real
            vmin = ( np.conj(grad)*elbPos ).real
            for ind in indices:
                if ind not in assignedTar: continue
                    
                tPos = self.targets.positions[ind]
                # screen out points outside arm space
                val = ( np.conj(grad)*tPos ).real
                if val>vmax or val < vmin: continue
                
                # distance to arm
                dist = np.abs( (np.conj(grad*1j)*(tPos - midPos)).real ) 
                if dist < COBRA_LINK_RADIUS*2+0.01:
                    collidedTar.add(ind)
                    collidedCob.add(cob)
        return collidedCob

    def selectTargetsByAuction(self):
        # build prices & unsettled_cobras
        targets = (np.unique(self.accessibleTargetIndices)[1:]).tolist() # ignore target -1
        priority = {i:self.targets.priorities[i] for i in targets}
        prices = {tar:0.0 for tar in targets}
        unset_targets = set(targets)
        unset_cobras = Queue()
        for i in range(self.bench.cobras.nCobras):
            unset_cobras.put(i)
        
        # Auction
        tar2cob = dict()
        self.assignedTargetIndices = np.full(self.bench.cobras.nCobras, NULL_TARGET_INDEX) # initalize by -1
        while unset_cobras.qsize()>0 and len(unset_targets)>0:
            cob = unset_cobras.get()
            items = []
            for tar in self.accessibleTargetIndices[cob]:
                if tar == -1: break
                items.append( (tar, priority[tar]-prices[tar]) )
            if len(items)==0: continue
            
            # update price-related variables
            if len(items)>=2:
                items = sorted(items, key=lambda x: -x[1]) # sort by descending marginal revenue
                delta_price = (items[0][1] - items[1][1])+0.001
            else: delta_price = np.inf
            if items[0][1] <0: continue
            
            # update target-related variables
            tar = items[0][0]
            prices[tar] += delta_price
            if tar in unset_targets: unset_targets.remove(tar)

            # update cobra-related variables
            if tar in tar2cob:
                unset_cobras.put(tar2cob[tar])
                self.assignedTargetIndices[tar2cob[tar]] = NULL_TARGET_INDEX
            tar2cob[tar] = cob
            self.assignedTargetIndices[cob] = tar

    def selectTargetsBySafeAuction(self):
        # build targets and target search tree
        targets = (np.unique(self.accessibleTargetIndices)[1:]).tolist() # ignore target -1
        priority = {i:self.targets.priorities[i] for i in targets}
        poss = self.targets.positions[targets]
        tree = cKDTree(np.column_stack((poss.real, poss.imag)))
        tar2tid = {tar:id for id, tar in enumerate(targets)}
        
        # build prices & unsettled_cobras
        prices = {tar:0.0 for tar in targets}
        unset_targets = set(targets)
        unset_cobras = Queue()
        for i in range(self.bench.cobras.nCobras):
            unset_cobras.put(i)
        
        # Auction
        tar2cob = dict()
        self.assignedTargetIndices = np.full(self.bench.cobras.nCobras, NULL_TARGET_INDEX) # initalize by -1
        while unset_cobras.qsize()>0 and len(unset_targets)>0:
            cob = unset_cobras.get()
            items = []
            for tar in self.accessibleTargetIndices[cob]:
                if tar == -1: break
                if tar not in unset_targets and tar not in tar2cob: continue # tar is within collision region
                if priority[tar]-prices[tar]<0: continue
                items.append( (tar, priority[tar]-prices[tar]) )
            if len(items)==0: continue
            
            # update price-related variables
            if len(items)>=2:
                items = sorted(items, key=lambda x: -x[1]) # sort by descending marginal revenue
                delta_price = (items[0][1] - items[1][1])+0.001
            else: delta_price = np.inf
            #if items[0][1] <0: continue
            
            # update target-related variables
            tar = items[0][0]
            prices[tar] += delta_price
            
            # update unset_targets and cobra-related variables
            if tar not in tar2cob:
                tg = tree.query_ball_point(tree.data[tar2tid[tar]], (COBRA_LINK_RADIUS*2)+0.01)
                for tid in tg:
                    if targets[tid] in unset_targets: 
                        unset_targets.remove(targets[tid])
            else:
                unset_cobras.put(tar2cob[tar])
                self.assignedTargetIndices[tar2cob[tar]] = NULL_TARGET_INDEX
            tar2cob[tar] = cob
            self.assignedTargetIndices[cob] = tar

    def selectTargetsByWeightedAuction(self, assiMode=2):
        # build targets and target search tree
        targets = (np.unique(self.accessibleTargetIndices)[1:]).tolist() # ignore target -1
        priority = {i:self.targets.priorities[i] for i in targets}
        poss = self.targets.positions[targets]
        tree = cKDTree(np.column_stack((poss.real, poss.imag)))
        tar2tid = {tar:id for id, tar in enumerate(targets)}
        
        # generate target neighborhood
        tarNei = dict()
        maxNeiSize = 0
        for tar in targets:
            tg = tree.query_ball_point(tree.data[tar2tid[tar]], (COBRA_LINK_RADIUS*2)+0.01)
            tarNei[tar] = [targets[tid] for tid in tg]
            if len(tg)>maxNeiSize: maxNeiSize=len(tg)
        
        # build prices & unsettled_cobras
        prices = {tar:0.0 for tar in targets}
        unset_targets = set(targets)
        unset_cobras = Queue()
        for i in range(self.bench.cobras.nCobras):
            unset_cobras.put(i)
        
        # Auction
        tar2cob = dict()
        self.assignedTargetIndices = np.full(self.bench.cobras.nCobras, NULL_TARGET_INDEX) # initalize by -1
        while unset_cobras.qsize()>0 and len(unset_targets)>0:
            cob = unset_cobras.get()
            items = []
            for tar in self.accessibleTargetIndices[cob]:
                if tar == -1: break
                if tar not in unset_targets and tar not in tar2cob: continue # tar is within collision region
                #if 1-prices[tar]<0: continue
                if assiMode==2:
                    items.append( (tar, (priority[tar]-prices[tar])/len(tarNei[tar])) )
                elif assiMode==3:
                    items.append( (tar, priority[tar]/len(tarNei[tar])-prices[tar] ) )
                elif assiMode==4:
                    items.append( (tar, priority[tar]- 0.01*len(tarNei[tar])/(maxNeiSize+1) - prices[tar] ) )
                else:
                    items.append( (tar, priority[tar]-prices[tar]) )
            if len(items)==0: continue
            
            # update price-related variables
            if len(items)>=2:
                items = sorted(items, key=lambda x: -x[1]) # sort by descending marginal revenue
                delta_price = (items[0][1] - items[1][1])+0.001
            else: delta_price = np.inf
            if items[0][1] <0: continue
            
            # update target-related variables
            tar = items[0][0]
            prices[tar] += delta_price
            
            # update unset_targets and cobra-related variables
            if tar not in tar2cob:
                for nei in tarNei[tar]:
                    if nei in unset_targets: 
                        unset_targets.remove(nei)
            else:
                unset_cobras.put(tar2cob[tar])
                self.assignedTargetIndices[tar2cob[tar]] = NULL_TARGET_INDEX
            tar2cob[tar] = cob
            self.assignedTargetIndices[cob] = tar

    def selectTargetsByGroupAuction(self):
        # build targets and target search tree
        targets = (np.unique(self.accessibleTargetIndices)[1:]).tolist() # ignore target -1
        poss = self.targets.positions[targets]
        tree = cKDTree(np.column_stack((poss.real, poss.imag)))
        tar2tid = {tar:id for id, tar in enumerate(targets)}
        
        # generate target neighborhood
        tarNei = dict()
        for tar in targets:
            tg = tree.query_ball_point(tree.data[tar2tid[tar]], (COBRA_LINK_RADIUS*2)+0.01)
            tarNei[tar] = [targets[tid] for tid in tg]
        
        # compute collided cobra and target
        def get_rm_cob_tar(tar): # note: require tar2cob
            rmCob = set()
            rmTar = set()
            for tt in tarNei[tar]:
                if tt in tar2cob:
                    cobidx = tar2cob[tt]
                    if cobidx not in rmCob:
                        rmCob.add(cobidx)
                        for ttt in cob2tars[cobidx]:
                            rmTar.add(ttt)
            return rmCob, rmTar
        
        # build prices & unsettled_cobras
        prices = {tar:0.01 for tar in targets}
        tar2cob = dict()
        cob2tars = dict()
        unset_cobras = Queue()
        for i in range(self.bench.cobras.nCobras):
            unset_cobras.put(i)
        
        # Auction
        self.assignedTargetIndices = np.full(self.bench.cobras.nCobras, NULL_TARGET_INDEX) # initalize by -1
        while unset_cobras.qsize()>0 and len(tar2cob)<len(targets):
            cob = unset_cobras.get()
            items = []
            for tar in self.accessibleTargetIndices[cob]:
                if tar == -1: break
                # compute group price
                _, tg = get_rm_cob_tar(tar)
                tg = tg | set(tarNei[tar])
                gprice = max([prices[tt] for tt in tg])
                
                if 1 - gprice < 0: continue
                else: items.append( (tar, 1-gprice) )
            if len(items)==0: continue
            
            # update price-related variables
            if len(items)>=2:
                items = sorted(items, key=lambda x: -x[1]) # sort by descending marginal revenue
                delta_price = (items[0][1] - items[1][1]) + 0.01
            else: delta_price = 0.01
            print(delta_price, len(items), gprice)
            
            # update price and the cobras & targets to remove due to collision
            tar = items[0][0]
            for tt in tarNei[tar]:
                prices[tt] += delta_price#/len(tarNei[tar])
            
            rmCob, rmTar = get_rm_cob_tar(tar)
            for tt in rmTar:
                del tar2cob[tt]
            for cc in rmCob:
                unset_cobras.put(cc)
                self.assignedTargetIndices[cc] = NULL_TARGET_INDEX
                del cob2tars[cc]
                
            # add new cobra and targets back
            for tt in tarNei[tar]:
                tar2cob[tt] = cob
            cob2tars[cob] = tarNei[tar]
            self.assignedTargetIndices[cob] = tar
'''
### Hungarian Algorithm ### 
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
    def selectTargetsByHungarian(self):
        # build target & cobra maps
        num2tar = np.unique(self.accessibleTargetIndices)[1:] # ignore target -1
        tar2num = dict()
        for i in range(len(num2tar)):
            tar2num[num2tar[i]] = i
        
        num2cob = []
        cob2num = dict()
        for i in range(len(self.accessibleTargetIndices)):
            if self.accessibleTargetIndices[i,0] != -1:
                num2cob.append(i)
                cob2num[i] = len(num2cob)-1
        
        # Hungarian
        row = []
        col = []
        data = []
        for i in range(len(self.accessibleTargetIndices)):
            for j in range(len(self.accessibleTargetIndices[0])):
                tar = self.accessibleTargetIndices[i,j]
                if tar != -1:
                    row.append(cob2num[i])
                    col.append(tar2num[tar])
                    data.append(-1)
        costM = csr_matrix((data, (row,col)), shape = (len(num2cob), len(num2tar)) ).toarray()
        row_ind, col_ind = linear_sum_assignment(costM)
        
        assign = col_ind
        costs = np.array( costM[row_ind, col_ind] )
        
        # make assignment
        self.assignedTargetIndices = np.full(self.bench.cobras.nCobras, NULL_TARGET_INDEX) # initalize by -1
        for i in range(len(assign)):
            if costs[i] == -1: # assigned to a target
                self.assignedTargetIndices[num2cob[i]] = num2tar[assign[i]]
'''
