import numpy as np

from MPS import perform_np_SVD
from MPS import tensordot

class iMPS:
    '''
    This class represents an infinite matrix product state where the assumption
    is made that the encoded quantum state as well as the given matrix product
    operator and thus the Hamiltonian governing the system is translationally
    invariant with a unit cell of length 2. This class consists of two site
    tensors of the MPS and two site tensors of the MPO. These site tensors must
    be instances of np.ndarray.
    '''

    def __init__(self,d=None,D=None):
        '''
        Initialises the iMPS class. This is done by providing the used local
        state space dimension d as well as the used local bond dimension D. The
        two MPS as well as two MPO sites present in this class is then not yet
        set and have to be set at a later time.
        '''

        self.L = 2

        self.siteA = None
        self.siteB = None

        # index order: a_l | b_l | a'_l
        self.Ltensor = np.array([[[1.0]]])

        # index order: a_r | b_r | a'_r
        self.Rtensor = np.array([[[1.0]]])

        self.MPO_L = None
        self.MPO_R = None

        self._useReducedTensors = False

    def set_MPO_left(self,MPO_site):
        '''
        Set the left MPO site tensor of the unit cell. Index order must be:
        b_l | b_m | s'_l | s_l.
        '''

        self.MPO_L = MPO_site

    def set_MPO_right(self,MPO_site):
        '''
        Set the right MPO site tensor of the unit cell. Index order must be:
        b_m | b_r | s'_r | s_r.
        '''

        self.MPO_R = MPO_site

    def get_MPO_left(self):
        '''
        Returns the left MPO site tensor of the unit cell. Index order is:
        b_l | b_m | s'_l | s_l.
        '''

        return self.MPO_L

    def get_MPO_right(self):
        '''
        Returns the right MPO site tensor of the unit cell. Index order is:
        b_m | b_r | s'_r | s_r.
        '''

        return self.MPO_R

    def get_Ltensor(self):
        '''
        Returns the L tensor. Index order is: a_l | b_l | a'_l.
        '''

        return self.Ltensor

    def get_Rtensor(self):
        '''
        Returns the R tensor. Index order is: a_r | b_r | a'_r.
        '''

        return self.Rtensor

    def absorb_sites_into_LR_tensors(self):
        '''
        Absorbs the site tensors in the unit cell into the L/R tensors where
        the left site tensor gets absorbed into the L tensor and the right site
        tensor gets absorbed into the R tensor. The site tensors of the unit
        cell are afterwards set to None.
        '''

        if self.siteA is not None:
            self.L += 1

            # index order: s_l | a_l | a_m @ a_l | b_l | a'_l
            #           => s_l | a_m | b_l | a'_l
            self.Ltensor = tensordot(self.siteA,self.Ltensor,axes=([1],[0]))

            # index order: s_l | a_m | b_l | a'_l @ b_l | b_m | s'_l | s_l 
            #           => a_m | a'_l | b_m | s'_l
            self.Ltensor = tensordot(self.Ltensor,self.MPO_L,
                                     axes=([0,2],[0,3]))

            # index order: a_m | a'_l | b_m | s'_l @ s'_l | a'_m | a'_l
            #           => a_m | b_m | a'_m
            self.Ltensor = tensordot(self.Ltensor,self._dagger(self.siteA),
                                     axes=([1,3],[2,0]))

        if self.siteB is not None:
            self.L += 1

            # index order: s_r | a_m | a_r @ a_r | b_r | a'_r
            #           => s_r | a_m | b_r | a'_r
            self.Rtensor = tensordot(self.siteB,self.Rtensor,axes=([2],[0]))

            # index order: s_r | a_m | b_r | a'_r @ b_m | b_r | s'_r | s_r
            #           => a_m | a'_r | b_m | s'_r
            self.Rtensor = tensordot(self.Rtensor,self.MPO_R,
                                     axes=([0,2],[3,1]))

            # index order: a_m | a'_r | b_m | s'_r @ s'_r | a'_r | a'_m
            #           => a_m | b_m | a'_m
            self.Rtensor = tensordot(self.Rtensor,self._dagger(self.siteB),
                                     axes=([1,3],[1,0]))

        # set site tensors in unit cell to None
        self.siteA = None
        self.siteB = None

    def insert_twosite_tensor(self,tensor,D=None,max_truncation_error=None):
        '''
        Insert a two-site tensor into the middle of the iMPS. The old site
        tensors are absorbed into the L/R tensors for this. The given two-site
        tensor 'tensor' which must have index order: s_l | s_r | a_l | a_r,
        is then split into two one site tensors representing the two site
        tensors in the unit cell. The two-site tensor is split with a singular
        value decomposition. The local bond dimension 'D' describes the number
        of singular values to keep thus defining the new local bond dimension
        of the central bond. Alternatively, we may set a maximum truncation
        error we are willing to accept. Setting both values to None causes no
        singular values to be cut. Setting both values to not None causes 'D'
        to overrule 'max_truncation_error' if necessary to keep the local bond
        dimension from exceeding D.

        Returns the tuple (D_,trErr,U,S,V) with 'D_' being the new local bond
        dimension, 'trErr' the truncation error encountered during the
        calculation of the one-site tensors, 'U' the left site tensor, 'S' the
        singular values and 'V' the right site tensor.
        '''

        self.absorb_sites_into_LR_tensors()

        if tensor is None:
            return

        # split tensor with SVD

        # index order: s_l | s_r | a_l | a_r => s_l | a_l | s_r | a_r
        tensor = tensor.swapaxes(1,2)

        tensor_shape = tensor.shape

        # index order: s_l | a_l | s_r | a_r => (s_l | a_l) | (s_r | a_r)
        tensor = tensor.reshape([tensor.shape[0]*tensor.shape[1],
                                 tensor.shape[2]*tensor.shape[3]])

        # check normalisation condition
        # must be V to be equal to chain growing in naive mode.

        # U index order: (s_l | a_l) | a_m
        # V index order:  a_m | (s_r | a_r)
        D_,trErr,U,S,V = perform_np_SVD(tensor, D=D,
                                     max_truncation_error=max_truncation_error,
                                        normalize='V', return_='U|S|V')

        # index order: (s_l | a_l) | a_m => s_l | a_l | a_m
        U = U.reshape(tensor_shape[0],tensor_shape[1],D_)

        # index order: a_m | (s_r | a_r) => a_m | s_r | a_r
        V = V.reshape(D_,tensor_shape[2],tensor_shape[3])

        # index order: a_m | s_r | a_r => s_r | a_m | a_r
        V = V.swapaxes(0,1)

        # set new tensors
        self.siteA = U
        self.siteB = V

        return D_,trErr,U,S,V

    def _dagger(self,tensor):
        '''
        Conjugates a given matrix or tensor and swaps the last two dimensions.
        '''

        return np.conjugate(tensor.swapaxes(-1,-2))

    def insert_both_site_tensors(self,siteA,siteB):
        '''
        Insert two new site tensors into the infinite MPS. The old site tensors
        are absorbed into the L/R tensors for this. Here, siteA is the new left
        site tensor of the unit cell, while siteB is the new right site tensor
        of the unit cell. The index order of the given tensors must be:

        siteA index order: s_l | a_l | a_m
        siteB index order: s_r | a_m | a_r
        '''

        self.absorb_sites_into_LR_tensors()
        self.siteA = siteA
        self.siteB = siteB

    def enlarge(self,D):
        '''
        Enlarges the MPS to the specified local bond dimension D by padding it
        with zeros.
        '''

        # pad site A
        pad_axis1 = max(0,D - self.siteA.shape[1])
        pad_axis2 = max(0,D - self.siteA.shape[2])
        pad_tuple = ((0,0),(0,pad_axis1),(0,pad_axis2))
        self.siteA = np.pad(self.siteA, pad_tuple,'constant',
                            constant_values=0)

        # pad site B
        pad_axis1 = max(0,D - self.siteB.shape[1])
        pad_axis2 = max(0,D - self.siteB.shape[2])
        pad_tuple = ((0,0),(0,pad_axis1),(0,pad_axis2))
        self.siteB = np.pad(self.siteB, pad_tuple,'constant',
                            constant_values=0)

        # pad Ltensor
        pad_axis1 = max(0,D - self.Ltensor.shape[0])
        pad_axis2 = max(0,D - self.Ltensor.shape[2])
        pad_tuple = ((0,pad_axis1),(0,0),(0,pad_axis2))
        self.Ltensor = np.pad(self.Ltensor, pad_tuple,'constant',
                              constant_values=0)

        # pad Ltensor
        pad_axis1 = max(0,D - self.Rtensor.shape[0])
        pad_axis2 = max(0,D - self.Rtensor.shape[2])
        pad_tuple = ((0,pad_axis1),(0,0),(0,pad_axis2))
        self.Rtensor = np.pad(self.Rtensor, pad_tuple,'constant',
                              constant_values=0)
