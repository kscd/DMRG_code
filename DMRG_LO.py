import numpy as np
import scipy as sp
from MPS import tensordot

# import reducedTensor class if it exists.
# If not, only dense tensors can be used.
try:
    import reducedTensor as rt
    _reducedTensorExists = True
except ImportError:
    _reducedTensorExists = False


class DMRG_LO(sp.sparse.linalg.LinearOperator):
    '''
    This class may replace the effective Hamiltonian H in the optimisation
    process for the case that no symmetries are used. The main idea is to
    perform the matrix-vector product inevitably occuring in an iterative
    eigensolver on the level of the tensors as they occur during the DMRG and
    not on the abstract level of the effective Hamiltonian as a matrix with the
    previous site tensor shaped as a vector. By not going to the abstract
    level, different possibilities to perform the matrix-vector product emerge,
    a number of which are more efficent. This class provides the possibility to
    perform the matrix-vector multiplication in the most efficient way
    possible, which runs to the third power of the local bond dimension.
    Furthermore, the demand on the main memory is greatly reduced and lies in
    the order of the second power of the local bond dimension.
    '''

    def __init__(self,L,R,W,dtype=np.dtype(np.float64)):
        '''
        Initialises the class DMRG_LO. Accepts the tensors which have to be of
        type np.ndarray. Furthermore, the following index order are required:

        Index order for one-site DMRG:
        L index order: a_p-1 | b_p-1 | a'_p-1
        R index order: a_p | b_p | a'_p
        W index order: b_p-1 | b_p | s'_p | s_p

        Index order for two-site DMRG:
        L index order: a_p-1 | b_p-1 | a'_p-1
        R index order: a_p+1 | b_p+1 | a'_p+1
        W index order: b_p-1 | b_p+1 | s'_p | s_p | s'_p+1 | s_p+1

        The tensor L represents the part of the tensor network MPS-MPO-MPS left
        of the sites being optimised. The same is true for the tensor R but it
        represents everything right of the sites in question. The tensor W is
        the MPO site tensor at the corresponding site. IF two-site DMRG is
        used, W represents the MPO site tensors at both positions.

        This function returns an instance of the class DMRG_LO.
        '''

        # store tensors, passed by reference
        self.L = L
        self.R = R
        self.W = W

        # figure out if one-site or two-site DMRG
        self.two_site = True if W.ndim == 6 else False

        # store shapes and dtype
        self.d  = W.shape[2]
        self.DL  = L.shape[0]
        self.DR  = R.shape[0]
        self.dtype = dtype

        if self.two_site:
            self.shape = (self.d*self.d*self.DL*self.DR,
                          self.d*self.d*self.DL*self.DR)
            self.M_size = self.d*self.d*self.DL*self.DR
        else:
            self.shape = (self.d*self.DL*self.DR,self.d*self.DL*self.DR)
            self.M_size = self.d*self.DL*self.DR

    def round(self,digits):
        '''
        Rounds the L, R and W tensors to 'digits' digits. This is done in an
        attempt to circumvent that ARPACK sometimes does not converge for no
        apparent reason.
        '''

        self.L = np.round(self.L,digits)
        self.R = np.round(self.R,digits)
        self.W = np.round(self.W,digits)

    def add_noise(self,strength):
        '''
        Add random noise to the L, R and W tensors of order 'strength'. This is
        done in an attempt to circumfact that ARPACK sometimes does not
        converge for no apparent reason.
        '''

        self.L = self.L + (np.random.random(self.L.shape)-0.5)*strength
        self.R = self.R + (np.random.random(self.R.shape)-0.5)*strength
        self.W = self.W + (np.random.random(self.W.shape)-0.5)*strength

    def _matvec(self,M):
        '''
        Multiply M to the tensor network H which is made up of the tensors L, W
        and R known to the instance of this class. The argument M must be of
        type np.ndarray and may be a vector or tensor. For the case that M is
        a vector the implicitly assumed index order is (s_p | a_p-1 | a_p) for
        one-site DMRG and (s_p | s_p+1 | a_p-1 | a_p+1) for two-site DMRG. To
        perform the tensor network contraction, M will be reshaped accordingly.
        Alternatively, M may already be given as a tensor of corresponding
        shape. If M is a tensor and contains one more dimension than expected,
        it is treated as a collection of several tensors M with the last
        dimension being the index to differentiate them. In this case, each of
        these tensors will be multiplied to H. Returns an array of type
        np.ndarray with the same shape as M was provided with.
        '''

        M_ndim = M.ndim

        # reshape if M was given as vector (eigensolver gives M as vector)
        if M.ndim == 1:
            M = (M.reshape([self.d,self.d,self.DL,self.DR]) if self.two_site
                 else M.reshape([self.d,self.DL,self.DR]))

        # perform contraction (extra = possible additional dimension)
        if self.two_site:

            # index order: a_p-1 | b_p-1 | a'_p-1
            #            @ s_p | s_p+1 | a_p-1 | a_p+1 | extra
            #           => b_p-1 | a'_p-1 | s_p | s_p+1 | a_p+1 | extra
            aux = tensordot(self.L,M,   axes=(0,2))

            # index order: b_p-1 | b_p+1 | s'_p | s_p | s'_p+1 | s_p+1
            #            @ b_p-1 | a'_p-1 | s_p | s_p+1 | a_p+1 | extra
            #           => b_p+1 | s'_p | s'_p+1 | a'_p-1 | a_p+1 | extra
            aux = tensordot(self.W,aux, axes=((0,3,5),(0,2,3)))

            # index order: b_p+1 | s'_p | s'_p+1 | a'_p-1 | a_p+1 | extra
            #            @ a_p+1 | b_p+1 | a'_p+1
            #           => s'_p | s'_p+1 | a'_p-1 | extra | a'_p+1
            aux = tensordot(aux,self.R, axes=((0,4),(1,0)))

        else:

            # index order: a_p-1 | b_p-1 | a'_p-1 @ s_p | a_p-1 | a_p | extra
            #           => b_p-1 | a'_p-1 | s_p | a_p | extra
            aux = tensordot(self.L,M,   axes=(0,1))

            # index order: b_p-1 | b_p | s'_p | s_p
            #            @ b_p-1 | a'_p-1 | s_p | a_p | extra
            #           => b_p | s'_p | a'_p-1 | a_p | extra
            aux = tensordot(self.W,aux, axes=((0,3),(0,2)))

            # index order: b_p | s'_p | a'_p-1 | a_p | extra
            #            @ a_p | b_p | a'_p => s'_p | a'_p-1 | extra | a'_p
            aux = tensordot(aux,self.R, axes=((0,3),(1,0)))

        # correct axis order if M is a collection of tensors
        if (self.two_site and M.ndim==5) or (not self.two_site and M.ndim==4):
            # index order: ... | extra | a'_p(+1) => ... a'_p(+1) | extra
            return aux.swapaxes(-1,-2)

        # flatten result if M was provided as vector and return
        return aux.reshape(aux.size) if M_ndim == 1 else aux

class DMRG_LO_PROJECTION(sp.sparse.linalg.LinearOperator):
    '''
    This class may replace the effective Hamiltonian H in the optimisation
    process for the case that no symmetries are used. Furthermore, it is
    possible to set the energy of a number of predefined states to zero thus
    preventing them from being targeted by the DMRG under the assumption that
    at least one further eigenstates with negative energy remains. The main
    idea is to perform the matrix-vector product inevitably occuring in an
    iterative eigensolver on the level of the tensors as they occur during the
    DMRG and not on the abstract level of the effective Hamiltonian as a matrix
    with the previous site tensor shaped as a vector. By not going to the
    abstract level, different possibilities to perform the matrix-vector
    product emerge, a number of which are more efficent. This class provides
    the possibility to perform the matrix-vector multiplication in the most
    efficient way possible, which runs to the third power of the local bond
    dimension. Furthermore, the demand on the main memory is greatly reduced
    and lies in the order of the second power of the local bond dimension.

    This class builds on top of the class DMRG_LO but does not inherit from it.
    '''

    def __init__(self,dmrg_lo,m,dtype=np.dtype(np.float64)):
        '''
        Initialises the class DMRG_LO_PROJECTION. Takes an instance of the
        class DMRG_LO named 'dmrg_lo' which provides an effective way to
        perform part of the contraction process. The argument 'm' is of type
        np.ndarray and stores the tensor or tensors representing the tensor
        environments which, in turn, incorporate the MPSs whose energies should
        be set to zero. The index order of the tensor m is
        's_p | a_p-1 | a_p | extra' for one-site DMRG and
        's_p | s_p+1 | a_p-1 | a_p+1 | extra' for two-site DMRG. Here 'extra'
        is a possibly present additional dimension needed if more than tensor
        is given to 'm' and used to refer to the different tensors thus stored.

        This function returns an instance of the class DMRG_LO_PROJECTION.
        '''

        # store given arguments, passed by reference
        self.dmrg_lo = dmrg_lo
        self.m = m

        # set properties
        self.two_site = self.dmrg_lo.two_site
        self.d  = self.dmrg_lo.d
        self.DL = self.dmrg_lo.DL
        self.DR = self.dmrg_lo.DR
        self.dtype = dtype

        # do further preparations
        if self.two_site:
            self.shape = (self.d*self.d*self.DL*self.DR,
                          self.d*self.d*self.DL*self.DR)
        else:
            self.shape = (self.d*self.DL*self.DR,self.d*self.DL*self.DR)

        self.mult_ortho = (True if (self.two_site and m.ndim == 5) or
                           (not self.two_site and m.ndim == 4) else False)

        if self.two_site and self.mult_ortho:
            self.axes_list = (0,1,2,3)
        elif self.two_site and not self.mult_ortho:
            self.axes_list = (0,1,2,3)
        elif not self.two_site and self.mult_ortho:
            self.axes_list = (0,1,2)
        elif not self.two_site and not self.mult_ortho:
            self.axes_list = (0,1,2)

        # precalculate H|m> and <m|H|m>
        self.Hm = self.dmrg_lo._matvec(m).reshape(m.shape)
        self.mHm = tensordot(np.conj(m),self.Hm,axes=(self.axes_list,
                                                      self.axes_list))

    def round(self,digits):
        '''
        Rounds the L, R and W tensors stored in the instance of DMRG_LO to
        'digits' digits. This is done in an attempt to circumvent that ARPACK
        sometimes does not converge for no apparent reason.
        '''

        self.dmrg_lo.round(digits)

    def add_noise(self,strength):
        '''
        Add random noise to the L, R and W tensors stored in the instance of
        DMRG_LO of order 'strength'. This is done in an attempt to circumfact
        that ARPACK sometimes does not converge for no apparent reason.
        '''

        self.dmrg_lo.add_noise(strength)

    def _matvec(self,M):
        '''
        Multiply M to the tensor network H which is made up of the tensors L, W
        and R known to the instance of this class. The argument M must be of
        type np.ndarray and may be a vector or tensor. For the case that M is
        a vector the implicitly assumed index order is (s_p | a_p-1 | a_p) for
        one-site DMRG and (s_p | s_p+1 | a_p-1 | a_p+1) for two-site DMRG. To
        perform the tensor network contraction, M will be reshaped accordingly.
        Alternatively, M may already be given as a tensor of corresponding
        shape. Returns an array of type np.ndarray with only one-dimension and
        the implicitly assumed index order as outlined above.
        '''

        # reshape M into tensor
        if self.two_site:
            M = M.reshape([self.d,self.d,self.DL,self.DR])
        else:
            M = M.reshape([self.d,self.DL,self.DR])

        # perform the tensor network contractions and return result
        a = tensordot(np.conj(self.Hm),M,axes=(self.axes_list,self.axes_list))
        b = tensordot(np.conj(self.m) ,M,axes=(self.axes_list,self.axes_list))

        if self.mult_ortho:
            x = tensordot(self.mHm,b,axes=((1),(0))) - a
            return (self.dmrg_lo._matvec(M.flatten())
                    + ( tensordot(self.m,x,axes=((-1),(0)))
                        - tensordot(b,self.Hm,axes=((0),(-1))) ).flatten())
        else:
            return (self.dmrg_lo._matvec(M.flatten())
                    + (self.m*(self.mHm*b-a) - b*self.Hm).flatten())


class DMRG_LO_PROJECTION2(sp.sparse.linalg.LinearOperator):
    '''
    This class may replace the effective Hamiltonian H in the optimisation
    process for the case that no symmetries are used. Furthermore, it is
    possible to elevate the energy of a number of predefined states by a given
    value thus preventing them from being targeted by the DMRG under the
    assumption that all states below the target state in terms of energy have
    been raised to an energy larger than that of the target state. The main
    idea is to perform the matrix-vector product inevitably occuring in an
    iterative eigensolver on the level of the tensors as they occur during the
    DMRG and not on the abstract level of the effective Hamiltonian as a matrix
    with the previous site tensor shaped as a vector. By not going to the
    abstract level, different possibilities to perform the matrix-vector
    product emerge, a number of which are more efficent. This class provides
    the possibility to perform the matrix-vector multiplication in the most
    efficient way possible, which runs to the third power of the local bond
    dimension. Furthermore, the demand on the main memory is greatly reduced
    and lies in the order of the second power of the local bond dimension.

    This class builds on top of the class DMRG_LO but does not inherit from it.
    '''

    def __init__(self,dmrg_lo,m,Delta,dtype=np.dtype(np.float64)):
        '''
        Initialises the class DMRG_LO_PROJECTION2. eTakes an instance of the
        class DMRG_LO named 'dmrg_lo' which provides an effective way to
        perform part of the contraction process. The argument 'm' is of type
        np.ndarray and stores the tensor or tensors representing the tensor
        environments which, in turn, incorporate the MPSs whose energies should
        be increased. The index order of the tensor m is
        's_p | a_p-1 | a_p | extra' for one-site DMRG and
        's_p | s_p+1 | a_p-1 | a_p+1 | extra' for two-site DMRG. Here 'extra'
        is a possibly present additional dimension needed if more than tensor
        is given to 'm' and used to refer to the different tensors thus stored.

        This function returns an instance of the class DMRG_LO_PROJECTION2.
        '''

        # store given arguments, passed by reference
        self.dmrg_lo = dmrg_lo

        # set properties
        self.two_site = self.dmrg_lo.two_site
        self.d  = self.dmrg_lo.d
        self.DL = self.dmrg_lo.DL
        self.DR = self.dmrg_lo.DR
        self.dtype = dtype
        self.Delta = Delta

        # do further preparations
        if self.two_site:
            self.shape = (self.d*self.d*self.DL*self.DR,
                          self.d*self.d*self.DL*self.DR)
        else:
            self.shape = (self.d*self.DL*self.DR,self.d*self.DL*self.DR)

        self.mult_ortho = (True if (self.two_site and m.ndim == 5) or
                           (not self.two_site and m.ndim == 4) else False)

        if self.two_site and self.mult_ortho:
            self.axes_list = (0,1,2,3)
        elif self.two_site and not self.mult_ortho:
            self.axes_list = (0,1,2,3)
        elif not self.two_site and self.mult_ortho:
            self.axes_list = (0,1,2)
        elif not self.two_site and not self.mult_ortho:
            self.axes_list = (0,1,2)

        if self.mult_ortho:
            self.LNR = m.reshape(-1,m.shape[-1])
        else:
            self.LNR = m.flatten()

    def round(self,digits):
        '''
        Rounds the L, R and W tensors stored in the instance of DMRG_LO to
        'digits' digits. This is done in an attempt to circumvent that ARPACK
        sometimes does not converge for no apparent reason.
        '''

        self.dmrg_lo.round(digits)

    def add_noise(self,strength):
        '''
        Add random noise to the L, R and W tensors stored in the instance of
        DMRG_LO of order 'strength'. This is done in an attempt to circumfact
        that ARPACK sometimes does not converge for no apparent reason.
        '''

        self.dmrg_lo.add_noise(strength)

    def _matvec(self,M):
        '''
        Multiply M to the tensor network H which is made up of the tensors L, W
        and R known to the instance of this class. The argument M must be of
        type np.ndarray and may be a vector or tensor. For the case that M is
        a vector the implicitly assumed index order is (s_p | a_p-1 | a_p) for
        one-site DMRG and (s_p | s_p+1 | a_p-1 | a_p+1) for two-site DMRG. To
        perform the tensor network contraction, M will be reshaped accordingly.
        Alternatively, M may already be given as a tensor of corresponding
        shape. Returns an array of type np.ndarray with only one-dimension and
        the implicitly assumed index order as outlined above.
        '''

        if self.mult_ortho:
            overlap = np.tensordot(self.LNR,M,axes=(0,0))
            return (self.dmrg_lo._matvec(M)
                    + self.Delta*np.tensordot(overlap,np.conj(self.LNR),
                                              axes=(0,1)))
        else:
            overlap = np.dot(self.LNR,M)
            return (self.dmrg_lo._matvec(M)
                    + self.Delta*overlap*np.conj(self.LNR))


class DMRG_LO_REDUCED(sp.sparse.linalg.LinearOperator):
    '''
    This class may replace the effective Hamiltonian H in the optimisation
    process for the case that no symmetries are used. The main idea is to
    perform the matrix-vector product inevitably occuring in an iterative
    eigensolver on the level of the tensors as they occur during the DMRG and
    not on the abstract level of the effective Hamiltonian as a matrix with the
    previous site tensor shaped as a vector. By not going to the abstract
    level, different possibilities to perform the matrix-vector product emerge,
    a number of which are more efficent. This class provides the possibility to
    perform the matrix-vector multiplication in the most efficient way
    possible, which runs to the third power of the local bond dimension.
    Furthermore, the demand on the main memory is greatly reduced and lies in
    the order of the second power of the local bond dimension. Furthermore,
    this class has been optimised for the usage of reduced tensors.
    '''

    def __init__(self,L,R,W,M,dtype=np.dtype(np.float64)):

        # store tensors, passed by reference
        self.L = L
        self.R = R
        self.W = W

        # figure out if one-site or two-site DMRG
        self.two_site = True if W.ndim == 6 else False

        # do further preparations
        self.purified_sector_list_M  = list(M.sectors.keys())

        self.purified_sector_size_M  = [M.sectors[item].size for item in
                                        self.purified_sector_list_M]

        self.purified_sector_shape_M = [M.sectors[item].shape for item in
                                        self.purified_sector_list_M]

        self.purified_size_M = sum(self.purified_sector_size_M)

        self.shape = (self.purified_size_M,self.purified_size_M)

        self.M_pipe_dict = M.pipe_dict
        self.M_pipehighestID = M.pipehighestID
        self.M_pipeID = M.pipeID
        self.M_q_vectors = M.q_vectors
        self.M_q_signs = M.q_signs
        self.dtype = dtype

        
    def cut_dense_part_out_M(self,vector):
        '''
        The scipy eigensolver can only work with true numpy arrays. Because of
        this, this function cuts the relevant part out of a given sparse vector
        or tensor and returns it for scipy to process.
        '''

        pure_sector_M = np.zeros(self.purified_size_M)

        filling_dim = 0
        for i,subsector in enumerate(self.purified_sector_list_M):

            if vector.ndim == 1:
                subsector = (subsector,)

            sector_shape = self.purified_sector_size_M[i]
            if subsector in vector.sectors:

                pure_sector_M[filling_dim:filling_dim+sector_shape] = (
                    np.real_if_close(vector.sectors[subsector].flatten()))

            else:

                pure_sector_M[filling_dim:filling_dim+sector_shape] = np.zeros(
                    sector_shape)

            filling_dim += sector_shape

        return pure_sector_M

    def slice_dense_part_in_sparse_M(self,vector):
        '''
        The dense numpy array given by scipy is sliced into a sparse vector so
        that this class may process it efficiently.
        '''

        # slice ground state back into sub_sectors
        GSS = {}
        start_pos = 0

        for i in range(len(self.purified_sector_list_M)):

            new_name = self.purified_sector_list_M[i]
            size = self.purified_sector_size_M[i]
            shape = self.purified_sector_shape_M[i]

            GSS[new_name] = vector[start_pos:start_pos+size]
            GSS[new_name] = GSS[new_name].reshape(shape)
            start_pos += size

        # inherit properties from old site tensor
        pipe_dict = self.M_pipe_dict
        pipehighestID = self.M_pipehighestID
        pipeID = self.M_pipeID
        q_vectors = self.M_q_vectors
        q_signs = self.M_q_signs

        state = rt.reducedTensor(tensor=GSS,
                                 list_of_q=q_vectors,list_of_xi=q_signs,Q=0,
                                 sectors_given=True,pipeID=pipeID,
                                 pipehighestID=pipehighestID,
                                 pipe_dict=pipe_dict)

        return state

    def _matvec(self,M):
        '''
        Multiply M, a vector of type np.ndarray to the tensor network H which
        is made up of the tensors L, W and R which are of type rt.reducedTensor
        and knownn to the instance of this class. The implicitly assumed index
        order of M is (s_p | a_p-1 | a_p) for one-site DMRG and
        (s_p | s_p+1 | a_p-1 | a_p+1) for two-site DMRG. To perform the tensor
        network contraction, M will be reshaped accordingly. Afterwards, the
        ensuring tensor of type rt.reducedTensor is converted back into a
        vector of type rt.reducedtensor and returned.
        '''

        # make dense vector M sparse tensor M2
        M2 = self.slice_dense_part_in_sparse_M(M)

        # perform contraction
        if M2.ndim == 4:

            # index order: a_p-1 | b_p-1 | a'_p-1
            #            @ s_p | s_p+1 | a_p-1 | a_p+1
            #           => b_p-1 | a'_p-1 | s_p | s_p+1 | a_p+1
            aux = tensordot(self.L,M2,  axes=((0,),(2,)))

            # index order: b_p-1 | b_p+1 | s'_p | s_p | s'_p+1 | s_p+1
            #            @ b_p-1 | a'_p-1 | s_p | s_p+1 | a_p+1
            #           => b_p+1 | s'_p | s'_p+1 | a'_p-1 | a_p+1
            aux = tensordot(self.W,aux, axes=((0,3,5),(0,2,3)))

            # index order: b_p+1 | s'_p | s'_p+1 | a'_p-1 | a_p+1
            #            @ a_p+1 | b_p+1 | a'_p+1
            #           => s'_p | s'_p+1 | a'_p-1 | a'_p+1
            aux = tensordot(aux,self.R, axes=((0,4),(1,0)))

        elif M2.ndim == 3:

            # index order: a_p-1 | b_p-1 | a'_p-1 @ s_p | a_p-1 | a_p
            #           => b_p-1 | a'_p-1 | s_p | a_p
            aux = tensordot(self.L,M2,   axes=((0,),(1,)))

            # index order: b_p-1 | b_p | s'_p | s_p
            #            @ b_p-1 | a'_p-1 | s_p | a_p
            #           => b_p | s'_p | a'_p-1 | a_p
            aux = tensordot(self.W,aux, axes=((0,3),(0,2)))

            # index order: b_p | s'_p | a'_p-1 | a_p
            #            @ a_p | b_p | a'_p => s'_p | a'_p-1 | a'_p
            aux = tensordot(aux,self.R, axes=((0,3),(1,0)))

        # convert to vector of type rt.reducedTensor and return
        return self.cut_dense_part_out_M(aux)
