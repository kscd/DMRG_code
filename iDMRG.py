import numpy as np
import scipy as sp

import scipy.sparse.linalg as linalg
import MPS_system
import iMPS
import copy

from MPS import tensordot
from MPS import perform_np_SVD

class iDMRG:
    '''
    The class 'iDMRG' is a class performing the iDMRG calculation and holding
    all necessary data for it. In this sense, it is an abstract class because
    it does not represent something one can 'touch'. This class has the ability
    to perform several variants of the iDMRG including the naive infinite DMRG
    and the variant from McCulloch. The iDMRG class assumes a unit cell of
    length 2.
    '''

    def __init__(self,MPO_left,MPO_right):
        '''
        Initialises the iDMRG class but giving the two sites tensors of the
        matrix product operator to it. This are the two MPO site tensors of the
        unit cell. Furthermore, this class is initialised by performing the
        iDMRG step for 2 and 4 sites exactly. The index order of the given MPO
        site tensors is:

        MPO_left  index order: b_l | b_m | s'_l | s_l
        MPO_right index order: b_m | b_r | s'_r | s_r
        '''

        ## Initialise system by solving it exactly for 2 and 4 sites.

        # setting local state space dimension and local bond dimension
        d  = MPO_left.shape[2]
        Dw = MPO_left.shape[0]

        # solve system for 2 sites exactly (to get Sold)

        # index order: b_m | s'_l | s_l @ b_m | s'_r | s_r
        #           => s'_l | s_l | s'_r | s_r
        W = tensordot(MPO_left[-1],MPO_right[:,0],[[0],[0]])

        # index order: s'_l | s_l | s'_r | s_r => s'_l | s'_r | s_l | s_r
        W = W.swapaxes(1,2)

        # index order: s'_l | s'_r | s_l | s_r => (s'_l | s'_r) | (s_l | s_r)
        W = W.reshape(d**2,d**2)

        w,v = np.linalg.eigh(W)

        # index order: (s_l | s_r)
        M = v[:,0]
        w2 = w[0]

        # index order: (s_l | s_r) => s_l | s_r
        M = M.reshape(d,d)
        _,_,Sold = perform_np_SVD(M, normalize='', return_='S',
                                  normalize_to_unity=True)

        # solve system for 4 sites exactly

        # index order: b_l | s'_l | s_l @ b_l | b_m | s'_ml | s_ml
        #           => s'_l | s_l | b_m | s'_ml | s_ml
        W = tensordot(MPO_left[-1],MPO_right,[[0],[0]])

        # index order: s'_l | s_l | b_m | s'_ml | s_ml
        #            @ b_m | b_r | s'_mr | s_mr
        #           => s'_l | s_l | s'_ml | s_ml | b_r | s'_mr | s_mr
        W = tensordot(W,MPO_left,[[2],[0]])

        # index order: s'_l | s_l | s'_ml | s_ml | b_r | s'_mr | s_mr
        #            @ b_r | s'_r | s_r
        #           => s'_l | s_l | s'_ml | s_ml | s'_mr | s_mr | s'_r | s_r
        W = tensordot(W,MPO_right[:,0],[[4],[0]])

        # index order: s'_l | s_l | s'_ml | s_ml | s'_mr | s_mr | s'_r | s_r
        #           => s'_l | s'_ml | s_l | s_ml | s'_mr | s_mr | s'_r | s_r
        W = W.swapaxes(1,2)

        # index order: s'_l | s'_ml | s_l | s_ml | s'_mr | s_mr | s'_r | s_r
        #           => s'_l | s'_ml | s'_mr | s_ml | s_l | s_mr | s'_r | s_r
        W = W.swapaxes(2,4)

        # index order: s'_l | s'_ml | s'_mr | s_ml | s_l | s_mr | s'_r | s_r
        #           => s'_l | s'_ml | s'_mr | s'_r | s_l | s_mr | s_ml | s_r
        W = W.swapaxes(3,6)

        # index order: s'_l | s'_ml | s'_mr | s'_r | s_l | s_mr | s_ml | s_r
        #           => s'_l | s'_ml | s'_mr | s'_r | s_l | s_ml | s_mr | s_r
        W = W.swapaxes(5,6)

        # index order: s'_l | s'_ml | s'_mr | s'_r  |  s_l | s_ml | s_mr | s_r
        #          => (s'_l | s'_ml | s'_mr | s'_r) | (s_l | s_ml | s_mr | s_r)
        W = W.reshape(d**4,d**4)

        w,v = np.linalg.eigh(W)

        # index order: (s_l | s_ml | s_mr | s_r)
        M = v[:,0]
        w4 = w[0]

        # index order: (s_l | s_ml | s_mr | s_r) => s_l | (s_ml | s_mr | s_r)
        M = M.reshape(d,d**3)

        # A1 index order: s_l | a_l
        #  M index order: a_l | (s_ml | s_mr | s_r)
        _,_,A1,M = perform_np_SVD(M, normalize='', return_='U|SV',
                                  normalize_to_unity=True)

        # index order: a_l | (s_ml | s_mr | s_r) => (a_l | s_ml | s_mr) | s_r
        M = M.reshape(d**3,d)

        #  M index order: (a_l | s_ml | s_mr) | a_r
        # B1 index order: a_r | s_r
        _,_,M,B1 = perform_np_SVD(M, normalize='', return_='US|V',
                                  normalize_to_unity=True)

        # index order: a_r | s_r => s_r | a_r
        B1 = B1.swapaxes(0,1)

        # index order: (a_l | s_ml | s_mr) | a_r => (a_l | s_ml) | (s_mr | a_r)
        M = M.reshape(d**2,d**2)

        # A2 index order: (a_l | s_ml) | a_m
        # B2 index order: a_m | (s_mr | a_r)
        _,_,A2,S,B2 = perform_np_SVD(M, normalize='', return_='U|S|V',
                                     normalize_to_unity=True)

        # index order: (a_l | s_ml) | a_m => a_l | s_ml | a_m
        A2 = A2.reshape(d,d,d**2)

        # index order: a_l | s_ml | a_m => s_ml | a_l | a_m
        A2 = A2.swapaxes(0,1)

        # index order: a_m | (s_mr | a_r) => a_m | s_mr | a_r
        B2 = B2.reshape(d**2,d,d)

        # index order: a_m | s_mr | a_r => s_mr | a_m | a_r
        B2 = B2.swapaxes(0,1)


        # store old singular values
        self.Sold = np.diag(np.pad(Sold, (0,d), 'constant', constant_values=0))
        self.Sold_inv = np.divide(1,self.Sold, out=np.zeros_like(self.Sold),
                                  where=self.Sold!=0)

        # store current singular values
        self.S = np.diag(S)
        self.S[np.abs(self.S) < 1e-12 ] = 0

        # create iMPS system
        self.system = iMPS.iMPS()

        # absorb outermost site tensors into L/R tensors
        self.system.set_MPO_left(MPO_left[None,-1])
        self.system.set_MPO_right(MPO_right[:,None,0])
        self.system.insert_both_site_tensors(A1.reshape(d,1,d),
                                             B1.reshape(d,d,1))
        self.system.absorb_sites_into_LR_tensors()

        # set next two site tensors
        self.system.set_MPO_left(MPO_left)
        self.system.set_MPO_right(MPO_right)
        self.system.insert_both_site_tensors(A2,B2)

        # precalculate two-site MPO

        # index order: b_l | b_m | s'_l | s_l @ b_m | b_r | s'_r | s_r
        #           => b_l | s'_l | s_l | b_r | s'_r | s_r
        self.W = tensordot(MPO_left,MPO_right,[[1],[0]])

        # index order: b_l | s'_l | s_l | b_r | s'_r | s_r
        #           => b_l | b_r | s_l | s'_l | s'_r | s_r
        self.W = self.W.swapaxes(1,3)

        # index order: b_l | b_r | s_l | s'_l | s'_r | s_r
        #           => b_l | b_r | s'_l | s_l | s'_r | s_r
        self.W = self.W.swapaxes(2,3)

        # set 'length' of the system (number of enlargement steps)
        self.L = 4

        # set current local bond dimension
        self.D = d**2

    def infinite_DMRG_McCulloch(self,L,D=None,method='g'):
        '''
        Performs the infinite DMRG in the variant described by McCulloch for
        L/2 steps where L needs to be an integer. The argument 'D' is the local
        bond dimension used during this calculation. The size of the site
        tensors cannot change in this variant. The argument 'method' describes
        the precise method being used and the following values are possible:

        'g'     : Optimise toward ground state (traditional DMRG step).
        'h'     : Optimise toward highest energy eigenstate
                  (same as 'g' for -H).
        'gED'   : Same as 'g' but uses exact diagonalisation. Does not use the
                  old site tensor as initial guess.
        'hED'   : Same as 'h' but uses exact diagonalisation. Does not use the
                  old site tensor as initial guess.
        'DMRGX' : To perform a DMRG-X step. Does not return the ground state of
                  H but instead the eigenstate of H closest in terms of spatial
                  overlap to the old site tensor.

        Returns the fidelities F and E as well as the singular values S, Sold
        and Sold_inv.

        TODO: check if this function is really correct. The index order turn at
        some point nonsensical. This may still be correct but then there must
        be a valid reason to rename them or this is incorrect and this function
        does not work properly.
        '''

        self._enlarge_system(D)

        # index order: s_l | a_l | a_m
        A = self.system.siteA

        # index order: s_r | a_m | a_r
        B = self.system.siteB

        F = np.zeros([L//2]) # fidelity
        E = np.zeros([L//2])

        for i in range(L//2):

            # absorb site tensors into L/R tensors
            self.system.absorb_sites_into_LR_tensors()

            # index order: s'_l | s'_r | a'_l | a'_r  |  s_l | s_r | a_l | a_r
            H = self._build_effective_Hamiltonian(False)

            # obtain guess for eigensolver

            # change normalization on both sites

            #   Anew   index order: s_r | a_m | c_m
            # Lambda_R index order: c_m | a_r
            Anew, Lambda_R = self._change_normalization_on_left_site(self.S,B)

            # Lambda_L index order: a_l | d_m
            #   Bnew   index order: s_l | d_m | a_m
            Lambda_L, Bnew = self._change_normalization_on_right_site(A,self.S)

            # get guess

            # index order: c_m | a_r @ a_r | a_r => c_m | a_r
            M = np.tensordot(Lambda_R,self.Sold_inv,axes=((1),(0)))

            # index order: c_m | a_r @ a_l | d_m => c_m | d_m
            M = np.tensordot(M,Lambda_L,axes=((1),(0)))

            # make s the s_old (s_old is the inverted old s)
            self.Sold = self.S
            self.Sold_inv = np.divide(1,self.Sold,
                                      out=np.zeros_like(self.Sold),
                                      where=self.Sold!=0)

            # update environment

            ##################################################
            # INDEX ORDER NONSENSICAL,                       #
            # MUST BE RENAMED AT SOME POINT OR IS INCORRECT. #
            ##################################################

            # index order: s'_l | s'_r | a'_l | a'_r | s_l | s_r | a_l | a_r
            #            @ s_r | a_m | c_m
            #           => s'_r | a'_r | s_l | s_r | a_l | a_r | c_m
            H = np.tensordot(H,Anew,axes=((0,2),(0,1)))

            # index order: s'_r | a'_r | s_l | s_r | a_l | a_r | c_m
            #            @ s_l | d_m | a_m
            #           => s_l | s_r | a_l | a_r | c_m | d_m
            H = np.tensordot(H,Bnew,axes=((0,1),(0,2)))

            # index order: s_l | s_r | a_l | a_r | c_m | d_m
            #            @ s'_r | c'_m | a'_m
            #           => s_r | a_r | c_m | d_m | c'_m
            H = np.tensordot(H,self._dagger(Anew),axes=((0,2),(0,2)))

            # index order: s_r | a_r | c_m | d_m | c'_m @ s'_l | a'_m | d'_m
            #           => c_m | d_m | c'_m | d'_m
            H = np.tensordot(H,self._dagger(Bnew),axes=((0,1),(0,1)))

            dimH = H.shape

            # index order:  c_m | d_m  |  c'_m | d'_m
            #           => (c_m | d_m) | (c'_m | d'_m)
            H = H.reshape(dimH[0]*dimH[1],dimH[2]*dimH[3])

            # measure fidelity

            # a_l | d_m @ a_m | a_m => a_l | a_m
            a = np.dot(Lambda_L,self._dagger(self.S))
            s = np.linalg.svd(a,compute_uv=False)
            F[i] = np.sum(s)

            # perform optimisation

            # index order: (c'_m | d'_m)
            w,M = self._optimisation_step(H,M,method=method)
            E[i] = w

            # A index order: c'_m | a''_m
            # B index order: a''_m | d'_m
            _,_,A,S,B = perform_np_SVD(M, D=D, normalize='S', return_='U|S|V',
                                       normalize_to_unity=True)

            # set S and modify it if necessary
            self.S = np.diag(S)

            # index order: s_r | a_m | c_m @ c'_m | a''_m => s_r | a_m | a''_m
            A = np.tensordot(Anew,A,axes=((2),(0)))

            # index order: a''_m | d'_m @ s_l | d_m | a_m => a''_m | s_l | a_m
            B = np.tensordot(B,Bnew,axes=((1),(1)))

            # index order: a''_m | s_l | a_m => s_l | a''_m | a_m
            B = B.swapaxes(0,1)

            self.system.siteA = A
            self.system.siteB = B

        return F,E,self.S,self.Sold,self.Sold_inv

    def infinite_DMRG_naive(self,L,D=None,max_truncation_error=None,
                            method='g'):
        '''
        Performs the infinite DMRG in the variant described by McCulloch for
        L/2 steps where L needs to be an integer. The argument 'D' is the local
        bond dimension used during this calculation. The size of the site
        tensors cannot change in this variant. The argument 'method' describes
        the precise method being used and the following values are possible:

        'g'     : Optimise toward ground state (traditional DMRG step).
        'h'     : Optimise toward highest energy eigenstate
                  (same as 'g' for -H).
        'gED'   : Same as 'g' but uses exact diagonalisation. Does not use the
                  old site tensor as initial guess.
        'hED'   : Same as 'h' but uses exact diagonalisation. Does not use the
                  old site tensor as initial guess.
        'DMRGX' : To perform a DMRG-X step. Does not return the ground state of
                  H but instead the eigenstate of H closest in terms of spatial
                  overlap to the old site tensor.

        Returns the truncation error, the fidelities F and E as well as the
        singular values S, Sold and Sold_inv.
        '''

        self._enlarge_system(D)

        # index order: s_l | a_l | a_m
        A = self.system.siteA

        # index order: s_r | a_m | a_r
        B = self.system.siteB

        trunc_err = np.zeros([L//2])
        F = np.zeros([L//2])
        E = np.zeros([L//2])

        for i in range(L//2):

            self.system.insert_twosite_tensor(None)

            H = self._build_effective_Hamiltonian()

            # obtain guess for eigensolver
            # calculate guess for next side: s v s_old u s

            # index order: a_m | a_m @ s_r | a_m | a_r => a_m | s_r | a_r
            M = np.tensordot(self.S, B, axes=([1],[1]))

            # index order: a_m | s_r | a_r @ a_r | a_l => a_m | s_r | a_l
            M = np.tensordot(M, self.Sold_inv, axes=([2],[0]))

            # index order: a_m | s_r | a_l @ s_l | a_l | a_m
            #           => a_m | s_r | s_l | a_m
            M = np.tensordot(M, A, axes=([2],[1]))

            # index order: a_m | s_r | s_l | a_m @ a_m | a_m
            #           => a_m | s_r | s_l | a_m
            M = np.tensordot(M, self.S, axes=([3],[0]))

            # index order: a_m | s_r | s_l | a_m => s_r | a_m | s_l | a_m
            M = M.swapaxes(0,1)

            # index order: s_r | s_l | a_m | a_m
            M = M.swapaxes(1,2)

            # make s the s_old (s_old is the inverted old s)
            # index order: a_r | a_l
            self.Sold = self.S
            self.Sold_inv = np.divide(1,self.S,out=np.zeros_like(self.S),
                                      where=self.S!=0)

            # perform optimisation
            w,M = self._optimisation_step(H,M,method=method)
            E[i] = w

            D_,trErr,_,S,_ = self.system.insert_twosite_tensor(
                M,D,max_truncation_error)

            self.S = np.diag(S)

            dimS = min(self.S.size,self.Sold_inv.size)
            F[i] = np.max(np.abs(self.S[:dimS] - self.Sold[:dimS]))

            trunc_err[i] = trErr

        return trunc_err,F,E,self.S,self.Sold_inv,self.Sold

    def check_left_normalization(self):
        '''
        Returns a measure for the left-normalization of the right site.
        Normalization is ensured if the returned matrix is an identity matrix.
        '''

        # index order: c_m | a_r
        Lambda_R = self._change_normalization_on_left_site(
            self.S,self.system.siteB)[1]

        # index order c_m | a_r @ a_r | a_l => c_m | a_l
        P = np.dot(Lambda_R,self.Sold_inv)

        # index order a_l | c_m @ c_m | a_l => a_l | a_l
        V = np.dot(self._dagger(P),P)

        return V

    def check_right_normalization(self):
        '''
        Returns a measure for the right-normalization of the left site.
        Normalization is ensured if the returned matrix is an identity matrix.
        '''

        # index order: a_l | d_m
        Lambda_L = self._change_normalization_on_right_site(
            self.system.siteA,self.S)[0]

        # index order: a_r | a_l @ a_l | d_m => a_r | d_m
        Q = np.dot(self.Sold_inv,Lambda_L)

        # index order: a_r | d_m @ d_m | a_r => a_r | a_r
        V = np.dot(Q,self._dagger(Q))

        return V

    def normalise(self):
        '''
        Normalizes the iMPS by changing the two iMPS site tensors accordingly.
        '''

        # index order: a_l | a_l
        VL = self.check_left_normalization()

        # index order: a_r | a_r
        VR = self.check_right_normalization()

        # index order: a'_l | a_l
        X = perform_np_SVD(VL,return_='sV')[2]

        # index order: a_r | a'_r
        Y = perform_np_SVD(VR,return_='Us')[2]

        # index order: a'_l | a_l @ s_l | a_l | a_m => a'_l | s_l | a_m
        self.system.siteA = np.tensordot(X,self.system.siteA,axes=((1),(1)))

        # index order: a'_l | s_l | a_m => s_l | a'_l | a_m
        self.system.siteA = self.system.siteA.swapaxes(0,1)

        # index order: s_r | a_m | a_r @ a_r | a'_r => s_r | a_m | a'_r
        self.system.siteB = np.tensordot(self.system.siteB,Y,axes=((2),(0)))

        # index order: a'_l | a_l @ a_l | a_r => a'_l | a_r
        self.Sold = np.dot(X,self.Sold)

        # index order: a'_l | a_r @ a_r | a'_r => a'_l | a'_r
        self.Sold = np.dot(self.Sold,Y)

        # index order: a_r | a_l
        self.Sold_inv = np.linalg.inv(self.Sold)

    def calculate_eval(self,operator_string):
        '''
        Calculate the expectation value with the given operator_string. The
        operator string contains an arbitrary number of local operators which
        are applied to the iMPS. Only one copy of the operator string is
        applied to the iMPS. For all other sites, the assumption is made that
        no operator operates there. Returns the expectation value thus
        described.
        '''

        # get both A sites

        # index order: s_l | a_l | a_m
        A = self.system.siteA

        #    A2    index order: s_r | a_m | c_m
        # Lambda_R index order: c_m | a_r
        A2, Lambda_R = self._change_normalization_on_left_site(
            self.S,self.system.siteB)

        # index order: c_m | a_r @ a_r | a_l => c_m | a_l
        P  = np.dot(Lambda_R,self.Sold_inv)

        # index order: s_r | a_m | c_m @ c_m | a_l => s_r | a_m | a_l
        A2 = np.tensordot(A2,P,axes=((2),(0)))

        # enlarge operator_string if not divisible by two
        if len(operator_string) % 2 == 1:
            operator_string.append(None)

        # perform contractions

        # index order: a_l | a_l
        ev = np.eye(A.shape[1])
        for i,O in enumerate(operator_string):

            # index order case 1: s_l | a_l | a_m
            # index order case 2: s_r | a_m | a_l
            site = A if i % 2 == 0 else A2

            # index order case 1: a_l | a_l @ s_l | a_l | a_m
            #                  => a_l | s_l | a_m
            # index order case 2: a_m | a_m @ s_r | a_m | a_l
            #                  => a_m | s_r | a_l
            ev = np.tensordot(ev,site,axes=([0],[1]))

            if O is None:
                # index order case 1: a_l | s_l | a_m @ s_l | a_m | a_l
                #                  => a_m | a_m
                # index order case 2: a_m | s_r | a_l @ s_r | a_l | a_m
                #                  => a_l | a_l
                ev = np.tensordot(ev,self._dagger(site),axes=([0,1],[2,0]))
            else:
                # index order case 1: a_l | s_l | a_m @ a_m | a_m
                #                  => a_l | s_l | a_m
                # index order case 2: a_m | s_r | a_l @ a_l | a_l
                #                  => a_m | s_r | a_l
                ev = np.tensordot(ev,O,axes=([1],[0]))

                # index order case 1: a_l | s_l | a_m @ s_l | a_m | a_l
                #                  => a_m | a_m
                # index order case 1: a_m | s_r | a_l @ s_r | a_l | a_m
                #                  => a_l | a_l
                ev = np.tensordot(ev,self._dagger(site),axes=([0,2],[2,0]))

        # perform final contraction over Sold

        # index order: a_l | a_l @ a_l | a_r => a_l | a_r
        ev = np.tensordot(ev,self.Sold,axes=([0],[0]))

        # index order: a_l | a_r @ a_r | a_l => a_r | a_r
        ev = np.tensordot(ev,self._dagger(self.Sold),axes=([0],[1]))

        # calculate trace and return
        ev = np.trace(ev)
        return ev

    def calculate_overlap(self,idmrg_):
        '''
        Calculate the overlap between the iMPS stored in this instance of class
        iDMRG and the iMPS stored in the instance 'idmrg_' of the class iDMRG.
        The returned overlap is the overlap per site as overlaps of the entire
        and infinite MPSs would result only in 1 or 0 thus removing most
        information.
        '''

        # index order: s_l | a_l | a_m
        A = self.system.siteA

        #    A2    index order: s_r | a_m | c_m
        # Lambda_R index order: c_m | a_r
        A2, Lambda_R = self._change_normalization_on_left_site(
            self.S,self.system.siteB)

        # index order: c_m | a_r @ a_r | a_r => c_m | a_r
        P  = np.dot(Lambda_R,self.Sold_inv)

        # index order: s_r | a_m | c_m @ c_m | a_r => s_r | a_m | a_r
        A2 = np.tensordot(A2,P,axes=((2),(0)))

        # index order: s_l | a'_l | a'_m
        A_ = idmrg_.system.siteA

        #    A2_    index order: s_r | a'_m | c'_m
        # Lambda_R_ index order: c'_m | a'_r
        A2_, Lambda_R_ = idmrg_._change_normalization_on_left_site(
            idmrg_.S,idmrg_.system.siteB)

        # index order: c'_m | a'_r @ a'_r | a'_r => c_m | a'_r
        P_  = np.dot(Lambda_R_,idmrg_.Sold_inv)

        # index order: s_r | a'_m | c'_m @ c'_m | a'_r => s_r | a'_m | a'_r
        A2_ = np.tensordot(A2_,P_,axes=((2),(0)))

        # index order: s_l | a_l | a_m @ s_r | a_m | a_r
        #           => s_l | a_l | s_r | a_r
        D  = np.tensordot(A,A2,axes=((2),(1)))

        # index order: s_l | a'_m | a'_l @ s_r | a'_r | a'_m
        #           => s_l | a'_l | s_r | a'_r
        D_ = np.tensordot(self._dagger(A_),self._dagger(A2_),axes=((1),(2)))

        # index order: s_l | a_l | s_r | a_r @ s_l | a'_l | s_r | a'_r
        #           => a_l | a_r | a'_l | a'_r
        V = np.tensordot(D,D_,axes=((0,2),(0,2)))

        # index order: a_l | a_r | a'_l | a'_r => a_l | a'_l | a_r | a'_r 
        V = V.swapaxes(1,2)

        # index order: a_l | a'_l | a_r | a'_r  => (a_l | a'_l) | (a_r | a'_r)
        V = V.reshape(V.shape[0]*V.shape[1],V.shape[2]*V.shape[3])

        # calculate eigenvalue with largest absolute value and return
        w = np.max(np.abs(np.linalg.eigvals(V)))
        return np.abs(w)

    ##########################################################
    # internal functions not meant to be called from outside #
    ##########################################################

    def _change_normalization_on_left_site(self,S,B):
        '''
        Changes the normalization on the left site by taking the current
        singular values 'S' and the right site tensor 'B' and returning the
        tensors Anew and Lambda_R.

        The following index orders have to be present, respectively, are being
        returned:

        S        : a_m | a_m
        B        : s_r | a_m | a_r
        Anew     : s_r | a_m | c_m
        Lambda_R : c_m | a_r
        '''

        # index order: a_m | a_m @ s_r | a_m | a_r => a_m | s_r | a_r
        M = np.tensordot(S,B,axes=((1,),(1,)))
        M_shape = M.shape

        # index order: a_m | s_r | a_r => (a_m | s_r) | a_r
        M = M.reshape(M_shape[0]*M_shape[1],M_shape[2])

        #   Anew   index order: (a_m | s_r) | c_m
        # Lambda_R index order: c_m | a_r
        _,_,Anew,Lambda_R = perform_np_SVD(M, normalize='', return_='U|SV',
                                           normalize_to_unity=True)

        # index order: (a_m | s_r) | c_m => a_m | s_r | c_m
        Anew = Anew.reshape(M_shape[0],M_shape[1],Anew.shape[-1])

        # index order: a_m | s_r | c_m => s_r | a_m | c_m
        Anew = Anew.swapaxes(0,1)

        return Anew,Lambda_R

    def _change_normalization_on_right_site(self,A,S):
        '''
        Changes the normalization on the left site by taking the left site
        tensor 'A' and the current singular values 'S' and returning the
        tensors Lambda_L and Bnew.

        The following index orders have to be present, respectively, are being
        returned:

        A        : s_l | a_l | a_m
        S        : a_m | a_m
        Lambda_L : a_l | d_m
        Bnew     : s_l | d_m | a_m
        '''

        # index order: s_l | a_l | a_m @ a_m | a_m => s_l | a_l | a_m
        M = np.tensordot(A,S,axes=((2,),(0,)))

        # index order: s_l | a_l | a_m => a_l | s_l | a_m
        M = M.swapaxes(0,1)
        M_shape = M.shape

        # index order: a_l | s_l | a_m => a_l | (s_l | a_m)
        M = M.reshape(M_shape[0],M_shape[1]*M_shape[2])

        # Lambda_L index order: a_l | d_m
        #   Bnew   index order: d_m | (s_l | a_m)
        _,_,Lambda_L,Bnew = perform_np_SVD(M, normalize='', return_='US|V',
                                           normalize_to_unity=True)

        # index order: d_m | (s_l | a_m) => d_m | s_l | a_m
        Bnew = Bnew.reshape(Bnew.shape[0],M_shape[1],M_shape[2])

        # index order: d_m | s_l | a_m => s_l | d_m | a_m
        Bnew = Bnew.swapaxes(0,1)

        return Lambda_L,Bnew


    def _enlarge_system(self,D):
        '''
        Enlarge all tensors to accomodate the new local bond dimension 'D'. If
        'D' is lower than the current local bond dimension, a ValueError is
        raised. Enlarged are the iMPS site tensors as well as the L/R tensors
        by padding them with zeros. Furthermore, the stored lists of singular
        values from the current and last optimisation step are also enlarged by
        padding them with zeros.
        '''

        if D < self.D:
            raise ValueError('D must not go lower.')

        self.system.enlarge(D)

        self.S        = np.pad(self.S,
                               ((0,max(0,D - self.S.shape[0])),
                                (0,max(0,D - self.S.shape[0]))),
                               'constant', constant_values=0)

        self.Sold     = np.pad(self.Sold,
                               ((0,max(0,D - self.Sold.shape[0])),
                                (0,max(0,D - self.Sold.shape[0]))),
                               'constant', constant_values=0)

        self.Sold_inv = np.pad(self.Sold_inv,
                               ((0,max(0,D - self.Sold_inv.shape[0])),
                                (0,max(0,D - self.Sold_inv.shape[0]))),
                               'constant', constant_values=0)


    def _build_effective_Hamiltonian(self,shape_to_matrix=True):
        '''
        Builds the effective Hamiltonian for the optimisation step of the iDMRG
        out of the L/R tensors and the MPO two-site tensor. If the argument
        'shape_to_matrix' is set to True, the index order of the returned
        effective Hamiltonian is:
        (s'_l | s'_r | a'_l | a'_r) | (s_l | s_r | a_l | a_r)
        while it is s'_l | s'_r | a'_l | a'_r | s_l | s_r | a_l | a_r
        for shape_to_matrix=True.
        '''

        # build effective Hamiltonian

        # index order: a_l | b_l | a'_l
        l = self.system.get_Ltensor()

        # index order: a_r | b_r | a'_r
        r = self.system.get_Rtensor()

        # index order: b_l | b_r | s'_l | s_l | s'_r | s_r @ a_r | b_r | a'_r
        #           => b_l | s'_l | s_l | s'_r | s_r | a_r | a'_r
        H = tensordot(self.W,r,axes=([1],[1]))

        # index order: a_l | b_l | a'_l
        #            @ b_l | s'_l | s_l | s'_r | s_r | a_r | a'_r
        #           => a_l | a'_l | s'_l | s_l | s'_r | s_r | a_r | a'_r
        H = tensordot(l,H,axes=([1],[0]))

        # index order: a_l | a'_l | s'_l | s_l | s'_r | s_r | a_r | a'_r
        #           => s'_l | a'_l | a_l | s_l | s'_r | s_r | a_r | a'_r
        H = H.swapaxes(0,2)

        # index order: s'_l | a'_l | a_l | s_l | s'_r | s_r | a_r | a'_r
        #           => s'_l | s'_r | a_l | s_l | a'_l | s_r | a_r | a'_r
        H = H.swapaxes(1,4)

        # index order: s'_l | s'_r | a_l | s_l | a'_l | s_r | a_r | a'_r
        #           => s'_l | s'_r | a'_l | s_l | a_l | s_r | a_r | a'_r
        H = H.swapaxes(2,4)

        # index order: s'_l | s'_r | a'_l | s_l | a_l | s_r | a_r | a'_r
        #           => s'_l | s'_r | a'_l | a'_r | a_l | s_r | a_r | s_l
        H = H.swapaxes(3,7)

        # index order: s'_l | s'_r | a'_l | a'_r | a_l | s_r | a_r | s_l
        #           => s'_l | s'_r | a'_l | a'_r | s_l | s_r | a_r | a_l
        H = H.swapaxes(4,7)

        # index order: s'_l | s'_r | a'_l | a'_r | s_l | s_r | a_r | a_l
        #           => s'_l | s'_r | a'_l | a'_r | s_l | s_r | a_l | a_r
        H = H.swapaxes(6,7)

        if not shape_to_matrix:
            return H

        dimH = np.shape(H)
        # index order:  s'_l | s'_r | a'_l | a'_r  |  s_l | s_r | a_l | a_r
        #           => (s'_l | s'_r | a'_l | a'_r) | (s_l | s_r | a_l | a_r)
        if self.system._useReducedTensors:
            rt.combine_axes_for_reduced_tensor(H,-1,4,4)
            rt.combine_axes_for_reduced_tensor(H,+1,0,4)
        else:
            H = H.reshape(dimH[0]*dimH[1]*dimH[2]*dimH[3],
                          dimH[4]*dimH[5]*dimH[6]*dimH[7])

        return H


    def _optimisation_step(self,H,v0,method='g'):
        '''
        The optimisation step in which an eigenstate of the effective
        Hamiltonian 'H' is calculated according to the supplied method. Based
        on the value of 'method', the argument v0 serves either as an initial
        guess, influences the optimisation in another capacity or does not
        influence it at all.

        This function takes the following arguments:

        H      : The effective Hamiltonian. Must be of type np.ndarray.
                 The index order must be (s'_k|a'_k-1| a'_k)|(s_k|a_k-1|a_k)
                 for one-site DMRG and
                 (s'_k|s'_k+1|a'_k-1| a'_k)|(s_k|s_k+1|a_k-1|a_k)
                 for two-site DMRG.
        v0     : The tensor accompanying the effective Hamiltonian. Its type
                 must be np.ndarray. The index order of v0 must be
                 (s_k|a_k-1|a_k) for one-site DMRG and (s_k|s_k+1|a_k-1|a_k)
                 for two-site DMRG.                       
        method : The precise method used in the optimisation step.
                 Default is 'g'. Options are given below.

        The variable 'method' may be set to the following values:

        'g'     : Optimise toward ground state (traditional DMRG step).
        'h'     : Optimise toward highest energy eigenstate
                  (same as 'g' for -H).
        'gED'   : Same as 'g' but uses exact diagonalisation. Does not use v0
                  as initial guess. Can only by used if H is of type np.ndarray
                  or rt.reducedTensor.
        'hED'   : Same as 'h' but uses exact diagonalisation. Does not use v0
                  as initial guess. Can only by used if H is of type np.ndarray
                  or rt.reducedTensor.
        'DMRGX' : To perform a DMRG-X step. Does not return the ground state of
                  H but instead the eigenstate of H closest in terms of spatial
                  overlap to v0. Can only by used if H is of type np.ndarray or
                  rt.reducedTensor.

        This function returns the tuple (w,v) with w being the energy
        eigenvalue of the newly calculated eigenstate v of H. Here, 'v' will be
        of the same type than 'v0'.

        TODO: This function is in earlier version of
        DMRG.DMRG._optimisation_step. Instead of just copying the newer version
        here, the function DMRG.DMRG._optimisation_step should be moved out of
        the DMRG.DMRG class body and into the DMRG module body. Afterwards,
        this function should be deleted and replaced by the then-named
        DMRG._optimisation_step function which just has to be imported.
        The only problem is that the DMRG.DMRG._optimisation_step function
        needs to be modified accordingly to work outside the DMRG class body.
        '''

        if method == 'g' or method == 'h':
            # perform DMRG

            # reshape v0 into a vector
            dimM = np.shape(v0)
            if self.system._useReducedTensors:
                rt.combine_axes_for_reduced_tensor(v0,+1,0,len(dimM))
            else:
                v0 = v0.reshape(np.prod(dimM))

            which = 'SA' if method == 'g' else 'LA'

            sweep_tolerance = 0.0


            w,v = linalg.eigsh(A=H, v0=v0, k=1, which=which,
                               tol=sweep_tolerance,
                               sigma=None, maxiter=100*len(H), ncv=10)

            return w[0],v.reshape(dimM)

        elif method == 'gED' or method == 'hED':
            # same as 'g' or 'h' but instead an exact eigensolver is used

            dimM = np.shape(v0)

            w,v = sp.linalg.eigh(a=H, overwrite_a=True, eigvals=None,
                                 check_finite=False)

            if method == 'gED':
                return w[0],v[:,0].reshape(dimM)
            else:
                return w[-1],v[:,-1].reshape(dimM)

        elif method == 'DMRGX':
            # perform DMRG-X

            dimM = np.shape(v0)

            #w,v = np.linalg.eigh(H)
            w,v = sp.linalg.eigh(a=H, overwrite_a=True, eigvals=None,
                                 check_finite=False)

            # find the state with the highest overlap to the previous one
            v0 = np.conjugate(v0.reshape(np.prod(dimM)))
            overlap = np.abs(tensordot(v0,v,axes=[[0],[0]]))
            highest_overlap_index = np.argmax(overlap)

            return (w[highest_overlap_index],
                    v[:,highest_overlap_index].reshape(dimM))

    def _dagger(self,tensor):
        '''
        Conjugates a given matrix or tensor and swaps the last two dimensions.
        '''

        return np.conjugate(tensor.swapaxes(-1,-2))
