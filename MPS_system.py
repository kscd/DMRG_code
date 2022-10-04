import numpy as np
import scipy.sparse.linalg as linalg
from MPS import _give_convenient_numpy_float_complex_type
from MPS import *
from DMRG_LO import DMRG_LO,DMRG_LO_PROJECTION,DMRG_LO_PROJECTION2

# import reducedTensor class if it exists
# if not, only dense tensors can be used
try:
    import reducedTensor as rt
    _reducedTensorExists = True
except ImportError:
    _reducedTensorExists = False

##########################################################
# Internal functions not meant to be called from outside #
##########################################################

def _optimisation_step(H,v0,method='g',useReducedTensors=False,
                       sweep_tolerance=0.0):
    '''
    A function resembling a DMRG optimisation step. Takes a matrix representing
    the effective Hamiltonian which may either be of type np.ndarray or of type
    rt.reducedTensor and operates on it to optimise the given initial state v0
    which must be of the same type as 'H' but is either three-dimensional for
    the one-site DMRG or four-dimensional for the two-site DMRG. The precise
    operation to be performed during the optimisation is given by value the
    variable 'method' is set to. Whether reduced Tensors are given or not must
    be supplied by setting the variable 'useReducedTensors' accordingly. In
    addition, it is possible to provide a tolerance the iterative eigensolver
    is willing to accept.

    For one-site DMRG:
    H has following index order:  
    v0 has following index order: s_p | a_p-1 | a_p

    For two-site DMRG:
    H has following index order:  
    v0 has following index order: s_p | s_p+1 | a_p-1 | a_p+1

    The variable 'method' may be set to the following values:
    'g'     : Optimise toward ground state (traditional DMRG step).
    'h'     : Optimise toward highest energy eigenstate (same as 'g' for -H).
    'gED'   : Same as 'g' but uses exact diagonalisation. Does not use v0.
    'hED'   : Same as 'h' but uses exact diagonalisation. Does not use v0.
    'DMRGX' : To perform a DMRG-X step. Is not optimised toward ground state
              but toward eigenstate closest in terms of spatial overlap to v0.

    This function returns the list [w,v] where w is the energy of the
    eigenstate calculated and v is the eigenstate itself shaped to the same
    shape v0 has.
    '''

    # During performance, tensor v0 is flattened thus having index order:
    # one-site DMRG index order:  s_p | a_p-1 | a_p
    #                         => (s_p | a_p-1 | a_p)
    # two-site DMRG index order:  s_p | s_p+1 | a_p-1 | a_p+1
    #                         => (s_p | s_p+1 | a_p-1 | a_p+1)

    dimM = np.shape(v0)

    if method == 'g' or method == 'h':

        # perform DMRG

        which = 'SA' if method == 'g' else 'LA'

        # reshape v0 into a vector
        if useReducedTensors:

            rt.combine_axes_for_reduced_tensor(v0,+1,0,len(dimM))
            w,v=rt.groundstate_searcher_for_reduced_matrix(H,v0,method=method)
            rt.split_axis_for_reduced_tensor(v,0)
            return w,v

        else:

            v0 = v0.reshape(np.prod(dimM))
            w,v=linalg.eigsh(A=H, v0=v0, k=1, which=which, tol=sweep_tolerance,
                             sigma=None, maxiter=100*H.shape[0], ncv=10)
            return w[0],v.reshape(dimM)

    elif method == 'gED' or method == 'hED':

        # The same as 'g' or 'h' but with an exact eigensolver

        if useReducedTensors:
            rt.combine_axes_for_reduced_tensor(v0,+1,0,len(dimM))
            w,v=rt.groundstate_searcher_for_reduced_matrix(H,v0,method=method)
            rt.split_axis_for_reduced_tensor(v,0)
            return w,v

        w,v = sp.linalg.eigh(a=H,overwrite_a=True,eigvals=None,
                             check_finite=False)

        if method == 'gED':
            return w[0],v[:,0].reshape(dimM)
        else:
            return w[-1],v[:,-1].reshape(dimM)

    elif method == 'DMRGX':

        # perform DMRG-X

        if useReducedTensors:

            rt.combine_axes_for_reduced_tensor(v0,+1,0,len(dimM))
            w,v=rt.groundstate_searcher_for_reduced_matrix(H,v0,method=method)
            rt.split_axis_for_reduced_tensor(v,0)
            return w,v

        #w,v = np.linalg.eigh(H)
        w,v = sp.linalg.eigh(a=H,overwrite_a=True,eigvals=None,
                             check_finite=False)

        # find the state with the highest overlap to the previous one
        v0 = np.conjugate(v0.reshape(np.prod(dimM)))
        overlap = np.abs(tensordot(v0,v,axes=[[0],[0]]))
        highest_overlap_index = np.argmax(overlap)

        return (w[highest_overlap_index],
                v[:,highest_overlap_index].reshape(dimM))

def _modify_effective_Hamiltonian2(H,L,R,N,fast=False,Delta=10):
    '''
    Modifies the effective Hamiltonian H by raising the energy of the MPSs the
    resulting eigenstate should be orthogonal to by 'Delta'. This is done to
    ensure that subsequent DMRG sweeps employing the effective Hamiltonian this
    function returns do not target the states we wish to forego but instead the
    then-ground state of the system. If fast=False, this will be done in a
    dense fashion.

    Index order of H: 
    Index order of L: i | a_p-1 | a'_p-1
    Index order of R: i | a_p+1 | a'_p+1 
    Index order of N: i | s_p | s_p+1 | a_p-1 | a_p+1
    The index i refers to the state to be projected out (arbitrary many
    allowed).

    This function is currently programmed for two-site DMRG but also supports
    one-site DMRG if the code is changed accordingly.

    Returns the modified effective Hamiltonian either as a dense matrix
    (fast=False) or an instance of the class DMRG_LO_PROJECTION2 (fast=True).
    '''

    a = len(N)

    two_site = True

    # Fill O (O is declared in the loop)
    for i in range(a):

        if two_site:

            # index order: s_p | s_p+1 | a_p+1 | a_p-1
            M = np.conjugate(N[i]).swapaxes(-1,-2)

            # index order: s_p | s_p+1 | a_p+1 | a_p-1 @ a_p-1 | a'_p-1
            #           => s_p | s_p+1 | a_p+1 | a'_p-1
            M = tensordot(M,L[i],axes=((3),(0)))

            # index order: s_p | s_p+1 | a_p+1 | a'_p-1 @ a_p+1 | a'_p+1
            #           => s_p | s_p+1 | a'_p-1 | a'_p+1
            M = tensordot(M,R[i],axes=((2),(0)))

        else:

            # index order: s_p | a_p | a_p-1
            M = np.conjugate(N[i]).swapaxes(-1,-2)

            # index order: s_p | a_p | a_p-1 @ a_p-1 | a'_p-1
            #           => s_p | a_p | a'_p-1
            M = tensordot(M,L[i],axes=((2),(0)))

            # index order: s_p | a_p | a'_p-1 @ a_p | a'_p
            #           => s_p | a'_p-1 | a'_p
            M = tensordot(M,R[i],axes=((1),(0)))

        if i == 0 and a > 1:
            O = np.empty([*M.shape,a])
        elif a == 1:
            O = M
            break

        O[...,i] = M

    if fast:
        return DMRG_LO_PROJECTION2(H,O,Delta)
    else:

        sger = None
        if H.dtype == np.float32:
            sger = sp.linalg.blas.sger
        elif H.dtype == np.float64:
            sger = sp.linalg.blas.dger
        elif H.dtype == np.complex64:
            sger = sp.linalg.blas.cgerc
        elif H.dtype == np.complex128:
            sger = sp.linalg.blas.zgerc

        if sger is not None and H.flags.f_contiguous:
            sger(alpha=Delta,a=H,x=N,y=N,overwrite_a=1,
                 overwrite_x=0,overwrite_y=0)
        else:
            H += Delta*np.outer(N,np.conj(N))

        return H

def _modify_effective_Hamiltonian(H,L,R,N,fast=False):
    '''
    Modifies the effective Hamiltonian by projecting out the MPSs the resulting
    eigenstate should be orthogonal to. If fast=False, this will be done in a
    dense fashion. Currently no implementation for reducedTensors exists.

    H is the Hamiltonian to be modified with L and R the overlap L-R-tensors
    and N the site tensors of the MPS that are to be projected out of H.

    Condition: fast = False, number of orthogonal MPS = 1, H is of type
    np.ndarray, H.flags.f_contiguous = True
    Consequence: The projecting out will be done inplace to avoid unnecessary
    memory allocations. Otherwise an array of the shape of H is allocated in
    the process. This, however, is very inefficient.

    Condition: fast = True
    Consequence: A new Linear operator is created which, when called, runs at
    D^3. It has the same scaling in leading order as the original LO but a
    slightly higher scaling for the D^2 term.

    In this function, the energy of the MPS-to-be-projected-out is set to 0.
    This method is not very good and the quality of subsequent DMRG calculation
    depends on how far the new ground state energy differs from 0.

    It is generally a better approach to use '_modify_effective_Hamiltonian2'
    instead of this function.
    '''

    a = len(N)

    two_site = True

    # Fill N (N is declared in the loop)
    for i in range(a):

        if two_site:

            # index order: s_p | s_p+1 | a_p+1 | a_p-1
            M = np.conjugate(N[i]).swapaxes(-1,-2)

            # index order: s_p | s_p+1 | a_p+1 | a_p-1 @ a_p-1 | a'_p-1
            #           => s_p | s_p+1 | a_p+1 | a'_p-1
            M = tensordot(M,L[i],axes=((3),(0)))

            # index order: s_p | s_p+1 | a_p+1 | a'_p-1 @ a_p+1 | a'_p+1
            #           => s_p | s_p+1 | a'_p-1 | a'_p+1
            M = tensordot(M,R[i],axes=((2),(0)))

        else:

            # index order: s_p | a_p | a_p-1
            M = np.conjugate(N[i]).swapaxes(-1,-2)

            # index order: s_p | a_p | a_p-1 @ a_p-1 | a'_p-1
            #           => s_p | a_p | a'_p-1
            M = tensordot(M,L[i],axes=((2),(0)))

            # index order: s_p | a_p | a'_p-1 @ a_p | a'_p
            #           => s_p | a'_p-1 | a'_p
            M = tensordot(M,R[i],axes=((1),(0)))

        if i == 0 and a > 1:
            O = np.empty([*M.shape,a])
        elif a == 1:
            O = M
            break

        O[...,i] = M

    if fast:
        return DMRG_LO_PROJECTION(H,O)
    else:

        # for dense H in (n x n), optimised for speed, needs to create an
        # aux. array of size(H).
        # Flops: 6 a n^2 + 2 n^2 , n = D^2 * d^x

        # Project O out
        if a == 1:
            O = O.reshape(np.prod(O.shape))
        else:
            O = O.reshape(np.prod(O.shape[:-1]),O.shape[-1])
        A = -np.tensordot(H,O,axes=((1),(0)))

        if a == 1:

            sger = None
            if H.dtype == np.float32:
                sger = sp.linalg.blas.sger
            elif H.dtype == np.float64:
                sger = sp.linalg.blas.dger
            elif H.dtype == np.complex64:
                sger = sp.linalg.blas.cgerc
            elif H.dtype == np.complex128:
                sger = sp.linalg.blas.zgerc

            if sger is not None and H.flags.f_contiguous:
                sger(alpha=1,a=H,x=A,y=O,overwrite_a=1,
                     overwrite_x=0,overwrite_y=0)
                A = -np.tensordot(N,H,axes=((0),(0)))
                sger(alpha=1,a=H,x=O,y=A,overwrite_a=1,
                     overwrite_x=0,overwrite_y=0)

            else:
                # manual implementation
                H += np.outer(A,np.conj(O))
                A = -np.tensordot(O,H,axes=((0),(0)))
                H += np.outer(O,np.conj(A))
        else:
            H +=  np.dot(A,O.T)
            A  = -np.tensordot(O,H,axes=((0),(0)))
            H +=  np.dot(O,A.T)

    return H

#################################################
# Functions to create an instance of MPS_system #
#################################################

def MPS_system_from_infinite_DMRG(L,MPO,D,d=2,max_truncation_error=None,
                                  method='g',orthogonal_MPSs=[],fast=False,
                                  ortho2=False,orthoDelta=10,
                                  *,print_steps=False):
    '''
    Receive an instance of MPS_system by performing the chain growing process
    of the infinite DMRG with the MPO supplied. For this, the given MPO does
    not require spatial translation invariance.

    This function takes the following arguments:

    L                    : Length of the spin chain in question.
    MPO                  : The MPO used during the chain growing process.
    D                    : The local bond dimension used during the iDMRG.
    d                    : The local state space dimension (d=2 for spin-1/2).
    max_truncation_error : The maximum truncation error to be accepted during
                           the iDMRG. Default is None. If set to None, local
                           bond dimension will solely be regulated by
                           argument 'D'. Local bond dimension will also not
                           exceed the value set by D but may well be smaller.
    method               : The precise method used in the optimisation step.
                           Default is 'g'. Options are given below.
    orthogonal_MPS       : A list that contains all the MPSs, the newly created
                           MPS should be held orthogonal to. Default is [] thus
                           marking that this MPS should be held orthogonal to
                           no other MPSs.
    fast                 : Whether the D^4 scaling or the D^3 scaling should be
                           used. Default is false, indicating the D^4 method.
    ortho2               : If this MPS should be held orthogonal to other MPSs,
                           decides the precise mechanism to do so. Default is
                           False. False results in raising the energy of all
                           involved MPSs by the value of 'orthoDelta', while
                           setting it to True results of all these energies
                           being implicitly set to zero.
    orthoDelta           : The energy by which the MPSs given orthogonal_MPS
                           are raised. Only applicable if ortho2=False.
                           Default is 10.
    print_steps          : Whether progress should be printed to the standard
                           output. Default is False.

    The variable 'method' may be set to the following values:

    'g'     : Optimise toward ground state (traditional DMRG step).
    'h'     : Optimise toward highest energy eigenstate (same as 'g' for -H).
    'gED'   : Same as 'g' but uses exact diagonalisation. Does not use old site
              tensor as initial guess. Can only by used if H is of type
              np.ndarray or rt.reducedTensor.
    'hED'   : Same as 'h' but uses exact diagonalisation. Does not use old site
              tensor as initial guess. Can only by used if H is of type
              np.ndarray or rt.reducedTensor.
    'DMRGX' : To perform a DMRG-X step. Is not optimised toward ground state
              but toward eigenstate closest in terms of spatial overlap to old
              site tensor. Can only by used if H is of type np.ndarray or
              rt.reducedTensor.

    This function returns the following list:
    [system, trunc_err, energy, ex_H_sq, variance]

    The entries of that list are as follows:
    system    : The instance of MPS_system created in this function.
    trunc_err : A list containing the truncation errors reiceived during the
                truncation of the central bond after each optimisation step.
                Length of that list is equal to L/2.
    energy    : The energy of the MPS under the present MPO.
    ex_H_sq   : The expectation value <psi|H^2|psi>
    variance  : The variance of the MPS under the present MPO.
    '''

    # perform consistency checks:
    if fast == True and method in ['gED','hED','DMRGX']:
        raise ValueError("The option fast=True may not be used together with "
                         "method={}.".format(method))
    
    useMPSo = False if orthogonal_MPSs == [] else True

    if useMPSo:

        # index order: c_0 | a_0
        Lo = [np.array([[1.0]]) for _ in range(len(orthogonal_MPSs))]

        # index order: c_0 | a_0
        Ro = [np.array([[1.0]]) for _ in range(len(orthogonal_MPSs))]

    system = MPS_system.empty_MPS(d=d,D=None)

    system.MPO = []
    system.set_LR_driver('manual')
    system.set_local_bond_dimension(D)

    assert L%2 == 0, 'L must be divisible by two.'

    trunc_err = np.zeros([L//2])
    S_old = np.array([1.])

    for i in range(L//2):

        # build two-site MPO

        # index order: b_i-1 | b_i | s'_i | s_i @ b_i | b_i+1 | s'_i+1 | s_i+1
        #           => b_i-1 | s'_i | s_i | b_i+1 | s'_i+1 | s_i+1
        W = tensordot(MPO[i],MPO[-i-1],[[1],[0]])

        # index order: b_i-1 | s'_i | s_i | b_i+1 | s'_i+1 | s_i+1
        #           => b_i-1 | b_i+1 | s_i | s'_i | s'_i+1 | s_i+1
        W = W.swapaxes(1,3)

        # index order: b_i-1 | b_i+1 | s_i | s'_i | s'_i+1 | s_i+1
        #           => b_i-1 | b_i+1 | s'_i | s_i | s'_i+1 | s_i+1
        W = W.swapaxes(2,3)

        # build effective Hamiltonian

        # index order: a_i-1 | b_i-1 | a'_i-1
        l = system.get_L_tensor(-1)

        # index order: a_i+1 | b_i+1 | a'_i+1
        R = system.get_R_tensor(-1)

        if fast:

            type_ = _give_convenient_numpy_float_complex_type(l.dtype.type,
                                                              R.dtype.type)

            type_ = _give_convenient_numpy_float_complex_type(type_,
                                                              W.dtype.type)

            H = DMRG_LO(l,R,W,dtype=np.dtype(type_))
            dimH = (H.d,H.d,H.DL,H.DR,H.d,H.d,H.DL,H.DR)

        else:

            # index order: b_i-1 | b_i+1 | s'_i | s_i | s'_i+1 | s_i+1
            #            @ a_i+1 | b_i+1 | a'_i+1
            #           => b_i-1 | s'_i | s_i | s'_i+1 | s_i+1 | a_i+1 | a'_i+1
            H = tensordot(W,R,axes=([1],[1]))

            # io: a_i-1 | b_i-1 | a'_i-1
            #   @ b_i-1 | s'_i | s_i | s'_i+1 | s_i+1 | a_i+1 | a'_i+1
            #  => a_i-1 | a'_i-1 | s'_i | s_i | s'_i+1 | s_i+1 | a_i+1 | a'_i+1
            H = tensordot(l,H,axes=([1],[0]))

            # io: a_i-1 | a'_i-1 | s'_i | s_i | s'_i+1 | s_i+1 | a_i+1 | a'_i+1
            #  => s'_i | a'_i-1 | a_i-1 | s_i | s'_i+1 | s_i+1 | a_i+1 | a'_i+1
            H = H.swapaxes(0,2)

            # io: s'_i | a'_i-1 | a_i-1 | s_i | s'_i+1 | s_i+1 | a_i+1 | a'_i+1
            #  => s'_i | s'_i+1 | a_i-1 | s_i | a'_i-1 | s_i+1 | a_i+1 | a'_i+1
            H = H.swapaxes(1,4)

            # io: s'_i | s'_i+1 | a_i-1 | s_i | a'_i-1 | s_i+1 | a_i+1 | a'_i+1
            #  => s'_i | s'_i+1 | a'_i-1 | s_i | a_i-1 | s_i+1 | a_i+1 | a'_i+1
            H = H.swapaxes(2,4)

            # io: s'_i | s'_i+1 | a'_i-1 | s_i | a_i-1 | s_i+1 | a_i+1 | a'_i+1
            #  => s'_i | s'_i+1 | a'_i-1 | a'_i+1 | a_i-1 | s_i+1 | a_i+1 | s_i
            H = H.swapaxes(3,7)

            # io: s'_i | s'_i+1 | a'_i-1 | a'_i+1 | a_i-1 | s_i+1 | a_i+1 | s_i
            #  => s'_i | s'_i+1 | a'_i-1 | a'_i+1 | s_i | s_i+1 | a_i+1 | a_i-1
            H = H.swapaxes(4,7)

            # io: s'_i | s'_i+1 | a'_i-1 | a'_i+1 | s_i | s_i+1 | a_i+1 | a_i-1
            #  => s'_i | s'_i+1 | a'_i-1 | a'_i+1 | s_i | s_i+1 | a_i-1 | a_i+1
            H = H.swapaxes(6,7)

            #io:s'_i | s'_i+1 | a'_i-1 | a'_i+1 | s_i | s_i+1 | a_i-1 | a_i+1
            #=>(s'_i | s'_i+1 | a'_i-1 | a'_i+1)|(s_i | s_i+1 | a_i-1 | a_i+1)
            dimH = np.shape(H)
            if system._useReducedTensors:
                rt.combine_axes_for_reduced_tensor(H,-1,4,4)
                rt.combine_axes_for_reduced_tensor(H,+1,0,4)
            else:
                H = H.reshape(dimH[0]*dimH[1]*dimH[2]*dimH[3],
                              dimH[4]*dimH[5]*dimH[6]*dimH[7])


        # obtain guess for eigensolver
        if i == 0:

            # index order: s_i | s_i+1 | a_i-1 | a_i+1
            M = np.random.rand(dimH[0],dimH[1],dimH[2],dimH[3])

        else:

            #calculate guess for next site: s v s_old u s

            # index order: a''_i | a''_i @ s_i+1 | a''_i | a_i+1
            #           => a''_i | s_i+1 | a_i+1
            M = np.tensordot(np.diag(S), V, axes=([1],[1]))

            # index order: a''_i | s_i+1 | a_i+1 @ a_i+1 | a_i-1
            #           => a''_i | s_i+1 | a_i-1
            M = np.tensordot(M, np.diag(S_old), axes=([2],[0]))

            # index order: a''_i | s_i+1 | a_i-1 @ s_i | a_i-1 | a''_i
            #           => a''_i | s_i+1 | s_i | a''_i
            M = np.tensordot(M, U, axes=([2],[1]))

            # index order: a''_i | s_i+1 | s_i | a''_i @ a''_i | a''_i
            #           => a''_i | s_i+1 | s_i | a''_i
            M = np.tensordot(M, np.diag(S), axes=([3],[0]))

            # index order: a''_i | s_i+1 | s_i | a''_i
            #           => s_i+1 | a''_i | s_i | a''_i
            M = M.swapaxes(0,1)

            # index order: s_i+1 | a''_i | s_i | a''_i
            #           => s_i+1 | s_i | a''_i | a''_i
            M = M.swapaxes(1,2)

            # make s the s_old (s_old is the inverted old s)
            # index order: a_i+1 | a_i-1
            S_old = np.divide(1,S, out=np.zeros_like(S), where=S!=0)

        # perform optimisation
        if useMPSo:

            N = []

            for MPS in orthogonal_MPSs:

                # index order: s_i | c_i-1 | c_i
                NL = MPS.get_site(i)

                # index order: s_i+1 | c_i | c_i+1
                NR = MPS.get_site(-i-1)

                # index order: s_i | c_i-1 | c_i @ s_i+1 | c_i | c_i+1
                #           => s_i | c_i-1 | s_i+1 | c_i+1
                N.append(np.tensordot(NL,NR,axes=((2),(1))))

                # index order: s_i | c_i-1 | s_i+1 | c_i+1
                #           => s_i | s_i+1 | c_i-1 | c_i+1
                N[-1] = N[-1].swapaxes(1,2)

            if ortho2:
                H = _modify_effective_Hamiltonian2(H,Lo,Ro,N,fast=fast,
                                                   Delta=orthoDelta)
            else:
                H = _modify_effective_Hamiltonian(H,Lo,Ro,N,fast=fast)

        w,M = _optimisation_step(H,M,method=method)

        if i == (L//2)-1:

            D_here,trErr = system.insert_twosite_tensor(i,M,MPO[i],MPO[-i-1],
                                                        normalization='l',
                                                        D=D,
                                     max_truncation_error=max_truncation_error)

        else:

            # U index order: s_i | a_i-1 | a''_i
            # S index order: a''_i | a''_i
            # V index order: s_i+1 | a''_i | a_i+1
            D_here,trErr,U,S,V = system.insert_twosite_tensor(
                i,M,MPO[i],MPO[-i-1],normalization='b',D=D,
                max_truncation_error=max_truncation_error,return_USV=True)

        if useMPSo: # and i < (L//2)-1: # i.e. not the last sweep
            for k,MPS in enumerate(orthogonal_MPSs):

                # index order: s_i | c_i-1 | c_i
                NL = MPS.get_site(i)

                # index order: s_i+1 | c_i | c_i+1
                NR = MPS.get_site(-i-1)

                # index order: s_i | a_i-1 | a_i
                ML = system.get_site(i)

                # index order: s_i+1 | a_i | a_i+1
                MR = system.get_site(-i-1)

                # index order: c_i-1 | a_i-1 @ s_i | c_i-1 | c_i
                #           => a_i-1 | s_i | c_i
                Lo[k] = np.tensordot(Lo[k],NL,axes=((0),(1)))

                # index order: a_i-1 | s_i | c_i @ s_i | a_i-1 | a_i
                #           => c_i | a_i
                Lo[k] = np.tensordot(Lo[k],ML,axes=((0,1),(1,0)))

                # index order: c_i+1 | a_i+1 @ s_i+1 | c_i | c_i+1
                #           => a_i+1 | s_i+1 | c_i
                Ro[k] = np.tensordot(Ro[k],NR,axes=((0),(2)))

                # index order: a_i+1 | s_i+1 | c_i @ s_i+1 | a_i | a_i+1
                #           => c_i | a_i
                Ro[k] = np.tensordot(Ro[k],MR,axes=((0,1),(2,0)))

        trunc_err[i] = trErr

        system.set_L_tensor_border(i+1)
        system.set_R_tensor_border(i+1)

    variance, ex_H_sq, energy, = system.energy_variance(return_all=True)
    return system, trunc_err, energy, ex_H_sq, variance


def reduced_MPS_system_from_basis_state(d,D,spin_list,MPO):
    '''
    Create an instance of the class MPS_system from a given basis state in the
    computational basis, i.e. the basis of classical states, by using reduced
    tensors. The argument 'd' is the local state space dimension (d=2 for
    spin-1/2), while 'D' is the local bond dimension the resulting MPS will
    have at the end. ('D' is currently not being used. State will have D=1.) As
    a classical state, it would be sufficient to set D=1 but it may be
    beneficial to set D>1 with an example being the one-site DMRG. For D>1, the
    site tensors are padded with zeros. The precise basis state to be encoded
    into the MPS is given by the argument spin_list, which contains numbers
    ranging from 0 to d-1 indicating the direction in which the spin is
    pointing at this position. The length of the MPS is given by the length of
    spin_list. The last argument to be supplied is the MPO governing the
    system.
    '''

    if not _reducedTensorExists:
        raise ImportError("'reducedTensor' could not be imported.")

    if d > 2:
        raise NotImplementedError('Local state space dimensions exceeding d=2'
                                  ' cannot be handled by this function at this'
                                  ' time.')
    
    # prepare empty MPS
    system = MPS_system.empty_MPS(d,D)

    # set initial charge vectors
    q_sigma  = [-1,1]
    q_a_left = [0]

    # provide further preparations
    Sz = sum((np.array(spin_list)-0.5)*2)
    L  = len(spin_list)
    sectors = give_valid_charge_sectors(L,Sz)

    # loop through the spin chain
    for l,spin in enumerate(spin_list):

        sector_dict = {}
        left_q  = sectors[l]
        right_q = sectors[l+1]

        # identify all possible sectors and set them accordingly
        for i in left_q:
            for j in right_q:

                if spin == 0:
                    if -i+j == -1:
                        sector_dict[(+1,i,j)] = np.array([[[1.]]])
                    if -i+j == +1:
                        sector_dict[(-1,i,j)] = np.array([[[0.]]])

                else:
                    if -i+j == -1:
                        sector_dict[(+1,i,j)] = np.array([[[0.]]])
                    if -i+j == +1:
                        sector_dict[(-1,i,j)] = np.array([[[1.]]])

        # calculate right charge sector
        q_a_right = [q_a_left[0]-q_sigma[spin]]
        
        # append site tensor to MPS
        system._MPS.append(rt.reducedTensor(tensor=sector_dict,
                                            list_of_q=[q_sigma,sectors[l],
                                                       sectors[l+1]],
                                            list_of_xi=[1,-1,1],Q=0,
                                            sectors_given=True))

        # left charge vector of next tensor must match
        # right charge vector of this tensor
        q_a_left = q_a_right

    # set properties of the MPS
    system._L = len(spin_list)
    system._B_border = len(spin_list)
    system._useReducedTensors = True

    # set MPO
    system.set_MPO(MPO)

    # set L and R dummy tensors
    q_L0 = system._MPS[0].q_vectors[1][0]
    q_L1 = system._MPO[0].q_vectors[0][0]

    q_R0 = system._MPS[-1].q_vectors[2][0]
    q_R1 = system._MPO[-1].q_vectors[1][0]

    # index order: a_0 | b_0 | a'_0
    system._L_tensors=[rt.reducedTensor({(q_L0,q_L1,q_L0):np.array([[[1.0]]])},
                                        [(q_L0,),(q_L1,),(q_L0,)],[1,1,-1],
                                        sectors_given=True)]

    # index order: a_L | b_L | a'_L
    system._R_tensors=[rt.reducedTensor({(q_R0,q_R1,q_R0):np.array([[[1.0]]])},
                                        [(q_R0,),(q_R1,),(q_R0,)],[-1,-1,1],
                                        sectors_given=True)]

    # return the instance of MPS_system
    return system

def load_MPS_system_from_MPS_hdf5(hdf5_handler,MPO):
    '''
    Load a saved MPS as an instance of the class MPS_system by applying a given
    MPS to the MPS. The MPS is loaded from a given hdf5_handler (from h5py).
    The handler must point to the group under whose name the instance
    originally got saved.
    '''

    # create empty instance of MPS_system
    MPS_ = MPS_system.empty_MPS(hdf5_handler.attrs['_d'],
                                hdf5_handler.attrs['_D'])

    # load attributes
    MPS_._L = hdf5_handler.attrs['_L']
    MPS_._multiplicative_factor = hdf5_handler.attrs['_multiplicative_factor']
    MPS_._normalisation_tolerance=(hdf5_handler.
                                   attrs['_normalisation_tolerance'])
    MPS_.initial_truncation_error=(hdf5_handler.
                                   attrs['initial_truncation_error'])
    MPS_._useReducedTensors = hdf5_handler.attrs['_useReducedTensors']
    MPS_._A_border = hdf5_handler.attrs['_A_border']
    MPS_._B_border = hdf5_handler.attrs['_B_border']

    # load site tensors and create L/R dummy tensors
    if hdf5_handler.attrs['_useReducedTensors']:
        MPS_._MPS=[rt.load_reducedTensor_hdf5(hdf5_handler['site{}'.format(i)])
                   for i in range(hdf5_handler.attrs['_L'])]

        q_L0 = MPS_._MPS[0].q_vectors[1][0]
        q_L1 = MPO[0].q_vectors[0][0]

        q_R0 = MPS_._MPS[-1].q_vectors[2][0]
        q_R1 = MPO[-1].q_vectors[1][0]

        # index order: a_0 | b_0 | a'_0
        MPS_._L_tensors = [rt.reducedTensor({(q_L0,q_L1,q_L0):
                                             np.array([[[1.0]]])},
                                            [(q_L0,),(q_L1,),(q_L0,)],[1,1,-1],
                                            sectors_given=True)]

        # index order: a_L | b_L | a'_L
        MPS_._R_tensors = [rt.reducedTensor({(q_R0,q_R1,q_R0):
                                             np.array([[[1.0]]])},
                                           [(q_R0,),(q_R1,),(q_R0,)],[-1,-1,1],
                                            sectors_given=True)]

    else:

        MPS_._MPS = [np.array(hdf5_handler['site{}'.format(i)])
                     for i in range(hdf5_handler.attrs['_L'])]

        # index order: a_0 | b_0 | a'_0
        MPS_._L_tensors = [np.array([[[1.0]]])]

        # index order: a_L | b_L | a'_L
        MPS_._R_tensors = [np.array([[[1.0]]])]

    MPS_._LR_driver = 'none'
    MPS_._L_border  =  0
    MPS_._R_border  = -1

    # set MPO
    MPS_.set_MPO(MPO)

    # return the instance of MPS_system
    return MPS_

class MPS_system(MPS):
    '''
    This class is basically the same as the MPS with the difference that the
    MPO of the Hamiltonian is added here. This gives the MPS a physical
    meaning. The MPS class does nothing else than manages the mathematical
    object. The inclusion of the MPO comes with added support for the L and R
    tensors. Which are entirely managed here and are implemented to be handled
    automatical.
    '''

    # overwrite init of MPS to include the existence of L/R tensors and MPO
    def __init__(self,*arg,**kw):
        '''
        Initialise the MPS_system class. There are several ways to initialise
        this class based on individual needs. The precise form the
        initialisation takes is determined by the variable 'initialise' which
        may take the following values:
        'const'        : Every element of every site tensor is set to the
                         numeric value given by 'const'.
        'constant'     : Same as 'const'.
        'basis',       : The MPS is created as a basis state of the
                         computational basis, i.e. as a classical state. May be
                         written with D=1. The precise state is given by
                         'spin_list' which describes with zeros and ones
                         whether the spin at the respective position should
                         point down or up. The length of the MPS is determined
                         by the length of this list.
        'basis state'  : Same as 'basis.
        'basis_state'  : Same as 'basis.
        'vector'       : Create the MPS from a given state vector. Only viable
                         for short chain lengths.
        'state vector' : Same as 'vector'.
        'state_vector' : Same as 'vector'.
        'empty'        : An empty MPS is created consisting of zero sites.

        In addition, a number of additional variables are present, explained in
        the following:
        d        : The local state space dimension. d=2 for spin-1/2, d=3 for
                   spin-1, etc.
        L        : The length of the spin chain, if required.
        D        : The local bond dimension the MPS should be created with.
        boundary : Whether open boundary conditions 'OBC' or periodic boundary
                   conditions 'PBC' shall be used. The support for 'PBC' is
                   rudimentary. The support for 'OBC' is absolut.
        state_vector_truncation_error : If the MPS is to be created from a
                   state vector, this variable holds the trunctation error that
                   the user is willing to accept during the decomposition into
                   an MPS.
        '''

        super().__init__(*arg,**kw)
        self._L_tensors = []
        self._R_tensors = []
        self._MPO = None            # is set later and separately
        self._LR_driver = 'none'    # is set later and separately
        self._L_border = 0          # for the LR driver 'manual'
        self._R_border = -1         # for the LR driver 'manual'

    def _add_sub_magic_method(self,MPS,factor):
        '''
        The function to which the magic method functions __add__ and __sub__
        are wrappers to. Performs the addition or subtraction of the MPS with
        another MPS. Here, addition means the superposition of the two MPS with
        equal phase, while subtraction refers to a superposition where the
        phase of the second MPS is different by a factor of pi to the phase of
        the first MPS. This function returns the respective superposition of
        the two MPS as a new MPS instance without having changed any of the
        initial ones.
        '''

        # This function is an exact copy pf the class MPS counterpart. It would
        # be better to delete all the duplicated code here, call the MPS
        # counterpart via super._add_sub_magic_method and to add the MPO by
        # calling a type conversion function which has yet to be written.

        # initial checks
        if self._L != MPS._L:
            raise ValueError('Both MPS need to be of the same length.')

        if self._d != MPS._d:
            raise ValueError('Both MPS need to have the same '
                             'local state space dimension.')

        # create empty MPS and evaluate boundary conditions
        MPSsum = MPS_system.empty_MPS(self._d,self._D)

        openBC = (True if self.get_boundary_condition() == 'OBC'
                  and MPS.get_boundary_condition() == 'OBC' else False)

        # loop through site tensors and concatenate them accordingly
        for site in range(self._L):

            tensor1 = self.get_site(site)
            tensor2 =  MPS.get_site(site)

            # set local bond dimension
            if openBC and site == 0:
                D_left = 1
            else:
                D_left  = np.shape(tensor1)[1] + np.shape(tensor2)[1]

            if openBC and site == self._L - 1:
                D_right = 1
            else:
                D_right = np.shape(tensor1)[2] + np.shape(tensor2)[2]

            # set site tensor
            new_type = _give_convenient_numpy_float_complex_type(
                                         tensor1.dtype.type,tensor2.dtype.type)
            new_tensor = np.empty([self._d,D_left,D_right],dtype=new_type)

            for sigma in range(self._d):

                if openBC and site == 0:
                    new_tensor[sigma] = np.concatenate((tensor1[sigma],
                                                        tensor2[sigma]),axis=1)
                elif openBC and site == self._L - 1:
                    new_tensor[sigma] = np.concatenate((tensor1[sigma],
                                                        tensor2[sigma]),axis=0)
                else:
                    new_tensor[sigma] = sp.linalg.block_diag(tensor1[sigma],
                                                             tensor2[sigma])

            MPSsum.insert_site_tensor(site,new_tensor,self._MPO[site],
                                      different_shape=True)

        return MPSsum

    # set MPO and get MPO

    def set_MPO(self,MPO):
        '''
        Sets the MPO of this MPS_system. The supplied MPO is a list containing
        the site tensors of the MPO which are either of type np.ndarray or of
        type rt.reducedTensor. Furthermore, the position of the site tensor in
        the list corresponds to the position in the spin chain. The index order
        of the i-th site tensor is: b_i-1 | b_i | s_i | s'_i.
        '''

        # perform consistency check
        if len(MPO) != self.get_length():
            raise ValueError('The supplied MPO has the wrong number of site '
                          'tensors. It should be {}'.format(self.get_length())+
                          ' but is {}'.format(len(MPO)))

        # prepare for the loop
        D_right_from_tensor_left = 1
        if self._useReducedTensors:
            q1_from_tensor_left = [0]

        # loop through elements in MPO and perform
        # site-dependent consistency checks
        for i,tensor in enumerate(MPO):

            self._check_if_array(tensor)
            self._check_if_tensor_has_right_number_of_dimensions(tensor,4)

            if np.shape(tensor)[0] == D_right_from_tensor_left:
                D_right_from_tensor_left = np.shape(tensor)[1]
            else:
                raise ValueError("Dimensions of the site tensors don't match "
                            "at position {} in the list of tensors.".format(i))

            if self._useReducedTensors:
                failed = False
                failed = (failed or
                          list(tensor.q_vectors[0]) != q1_from_tensor_left)
                failed = failed or list(tensor.q_vectors[2]) != [-1,1]
                failed = failed or list(tensor.q_vectors[3]) != [-1,1]
                failed = failed or list(tensor.q_signs) != [-1,1,1,-1]

                if failed:
                    raise ValueError("Charge vectors or signs of the site "
                                     "tensors don't match at position "
                                     "{} in the list of tensors.".format(i))

                q1_from_tensor_left = tensor.q_vectors[1]

        if np.shape(MPO[-1])[1] != 1:
            raise ValueError("Dimensions of the site tensors don't match at "
                             "the last position in the list of tensors.")

        if self._useReducedTensors and q1_from_tensor_left != [0]:
            raise ValueError("Charge vector of the site tensors don't match "
                             "at the last position in the list of tensors.")

        # set tensors (did checks separately to prevent possible data loss)
        self._MPO = []
        for i,tensor in enumerate(MPO):

            # set site tensor
            self._MPO.append(tensor)

    def get_MPO_site(self,pos):
        '''
        Returns the four-dimensional site tensor of the MPO at position 'pos'.
        Index order: b_p-1 | b_p | s_p | s'_p
        '''

        self._check_if_position_is_valid(pos)
        return self._MPO[pos]

    def get_MPO_twosite_tensor(self,pos_left):
        '''
        Returns the combined site tensors for positions 'pos_left' and
        'pos_left+1'. This results in an 6 dimensional site tensor with index
        order: b_p-1 | b_p+1 | s'_p | s_p | s'_p+1 | s_p+1.
        '''

        # index order: b_p-1 | b_p | s'_p | s_p @ b_p | b_p+1 | s'_p+1 | s_p+1
        #           => b_p-1 | s'_p | s_p | b_p+1 | s'_p+1 | s_p+1
        combined_site_tensor = tensordot(self.get_MPO_site(pos_left),
                                       self.get_MPO_site(pos_left+1),[[1],[0]])

        # index order: b_p-1 | s'_p | s_p | b_p+1 | s'_p+1 | s_p+1
        #           => b_p-1 | b_p+1 | s_p | s'_p | s'_p+1 | s_p+1
        combined_site_tensor = combined_site_tensor.swapaxes(1,3)

        # index order: b_p-1 | b_p+1 | s_p | s'_p | s'_p+1 | s_p+1
        #           => b_p-1 | b_p+1 | s'_p | s_p | s'_p+1 | s_p+1
        combined_site_tensor = combined_site_tensor.swapaxes(2,3)
        return combined_site_tensor

    ###############################################
    # functions to manipulate the L and R tensors #
    ###############################################

    def get_LR_driver(self):
        '''
        Returns the LR driver. The returned string represents the following:
        'none'     : Deactivates the usage of the LR tensors. This saves time
                     if they are not needed.
        'complete' : Calculates the complete set of L tensors and R tensors and
                     updates them, thenever sites are changed.
        'manual'   : Only calculates the L and R tensors that are manually
                     requested. This saves time, if not all tensors are needed
                     as compared to 'complete'.
        'locked'   : Locks the tensors to the A- and B-boundaries. This is the
                     normal operation mode for the DMRG.
        '''

        return self._LR_driver

    def set_LR_driver(self,driver):
        '''
        Sets the LR driver. The argument 'driver' may be set to the following
        values:
        'none'     : Deactivates the usage of the LR tensors. This saves time
                     if they are not needed.
        'complete' : Calculates the complete set of L tensors and R tensors and
                     updates them, thenever sites are changed.
        'manual'   : Only calculates the L and R tensors that are manually
                     requested. This saves time, if not all tensors are needed
                     as compared to 'complete'.
        'locked'   : Locks the tensors to the A- and B-boundaries. This is the
                     normal operation mode for the DMRG.
        '''

        if driver == 'none':

            #delete all entries appart from the dummy entries.
            self._LR_driver = driver

            # L index order: a_0 | b_0 | a'_0
            # R index order: a_L | b_L | a'_L
            if self._useReducedTensors:

                q_L0 = self._MPS[0].q_vectors[1][0]
                q_L1 = self._MPO[0].q_vectors[0][0]
                self._L_tensors = [rt.reducedTensor({(q_L0,q_L1,q_L0):
                                                     np.array([[[1.0]]])},
                                                    [(q_L0,),(q_L1,),(q_L0,)],
                                                    [1,1,-1],
                                                    sectors_given=True)]

                q_R0 = self._MPS[-1].q_vectors[2][0]
                q_R1 = self._MPO[-1].q_vectors[1][0]
                self._R_tensors = [rt.reducedTensor({(q_R0,q_R1,q_R0):
                                                     np.array([[[1.0]]])},
                                                    [(q_R0,),(q_R1,),(q_R0,)],
                                                    [-1,-1,1],
                                                    sectors_given=True)]

            else:

                self._L_tensors = [np.array([[[1.0]]])]
                self._R_tensors = [np.array([[[1.0]]])]

        elif driver in ['complete','manual','locked']:

            self._LR_driver = driver
            self._recalculate_L_tensors_behind_pos(0)
            self._recalculate_R_tensors_before_pos(-1)

        else:

            raise ValueError("Driver '{}' not recognised. ".format(driver)+
                             "Available drivers are 'none', 'complete', "
                             "'manual' and 'locked'.".format(driver))

    def energy_variance(self,return_all=False):
        '''
        Calculates the energy variance <psi|H^2|psi> - <psi|H|psi>^2
        for the MPS with its given MPO. If return_all=False, returns the energy
        variance and the energy <psi|H|psi>. For return_all=True, returns the
        energy variance, the expectation value <psi|H^2|psi> as well as the
        energy <psi|H|psi>.
        '''

        L = self.get_length()

        # perform first contraction thus creating the auxiliary tensors N and O
        
        # index order: s_1 | a_0 | a_1
        M = self.get_site(0)

        # index order: b_0 | b_1 | s'_1 | s_1
        W = self.get_MPO_site(0)


        # index order: s_1 | a_0 | a_1 @ b_0 | b_1 | s'_1 | s_1
        #           => a_0 | a_1 | b_0 | b_1 | s'_1
        N = tensordot(M,W,axes=([0],[3]))

        # index order: a_0 | a_1 | b_0 | b_1 | s'_1 @ s'_1 | a'_1 | a'_0
        #           => a_0 | a_1 | b_0 | b_1 | a'_1 | a'_0
        O = tensordot(N,self._dagger(M),axes=([4],[0]))

        # index order: a_0 | a_1 | b_0 | b_1 | s'_1
        #            @ b'_0 | b'_1 | s''_1 | s'_1
        #           => a_0 | a_1 | b_0 | b_1 | b'_0 | b'_1 | s''_1
        N = tensordot(N,W,axes=([4],[3]))

        # index order: a_0 | a_1 | b_0 | b_1 | b'_0 | b'_1 | s''_1
        #            @ s''_1 | a'_1 | a'_0
        #           => a_0 | a_1 | b_0 | b_1 | b'_0 | b'_1 | a'_1 | a'_0
        N = tensordot(N,self._dagger(M),axes=([6],[0]))


        # index order: a_0 | a_1 | b_0 | b_1 | a'_1 | a'_0
        #           => a_0 | b_0 | a_1 | b_1 | a'_1 | a'_0
        O = O.swapaxes(1,2)

        # index order: a_0 | a_1 | b_0 | b_1 | a'_1 | a'_0
        #           => a_0 | b_0 | a'_0 | b_1 | a'_1 | a_1
        O = O.swapaxes(2,5)

        # index order: a_0 | b_0 | a'_0 | b_1 | a'_1 | a_1
        #           => a_0 | b_0 | a'_0 | a_1 | a'_1 | b_1
        O = O.swapaxes(3,5)

        # index order: a_0 | b_0 | a'_0 | a_1 | a'_1 | b_1
        #           => a_0 | b_0 | a'_0 | a_1 | b_1 | a'_1
        O = O.swapaxes(4,5)


        # index order: a_0 | a_1 | b_0 | b_1 | b'_0 | b'_1 | a'_1 | a'_0
        #           => a_0 | b_0 | a_1 | b_1 | b'_0 | b'_1 | a'_1 | a'_0
        N = N.swapaxes(1,2)

        # index order: a_0 | b_0 | a_1 | b_1 | b'_0 | b'_1 | a'_1 | a'_0
        #           => a_0 | b_0 | b'_0 | b_1 | a_1 | b'_1 | a'_1 | a'_0
        N = N.swapaxes(2,4)

        # index order: a_0 | b_0 | b'_0 | b_1 | a_1 | b'_1 | a'_1 | a'_0
        #           => a_0 | b_0 | b'_0 | a'_0 | a_1 | b'_1 | a'_1 | b_1
        N = N.swapaxes(3,7)

        # index order: a_0 | b_0 | b'_0 | a'_0 | a_1 | b'_1 | a'_1 | b_1
        #           => a_0 | b_0 | b'_0 | a'_0 | a_1 | b_1 | a'_1 | b'_1
        N = N.swapaxes(5,7)

        # index order: a_0 | b_0 | b'_0 | a'_0 | a_1 | b_1 | a'_1 | b'_1
        #           => a_0 | b_0 | b'_0 | a'_0 | a_1 | b_1 | b'_1 | a'_1
        N = N.swapaxes(6,7)

        # loop over rest of the tensor networks
        for i in range(1,L):

            # index order: s_p | a_p-1 | a_p
            M = self.get_site(i)

            # index order: b_p-1 | b_p | s'_p | s_p
            W = self.get_MPO_site(i)

            # io: a_0 | b_0 | b'_0 | a'_0 | a_p-1 | b_p-1 | b'_p-1 | a'_p-1
            #   @ s_p | a_p-1 | a_p
            #  => a_0 | b_0 | b'_0 | a'_0 | b_p-1 | b'_p-1 | a'_p-1 | s_p | a_p
            N = tensordot(N,M,axes=([4],[1]))

            # io: a_0 | b_0 | b'_0 | a'_0 | b_p-1 | b'_p-1 | a'_p-1 | s_p | a_p
            #   @ b_p-1 | b_p | s'_p | s_p
            #  => a_0 | b_0 | b'_0 | a'_0 | b'_p-1 | a'_p-1 | a_p | b_p | s'_p
            N = tensordot(N,W,axes=([4,7],[0,3]))

            # io: a_0 | b_0 | b'_0 | a'_0 | b'_p-1 | a'_p-1 | a_p | b_p | s'_p
            #   @ b'_p-1 | b'_p | s''_p | s'_p
            #  => a_0 | b_0 | b'_0 | a'_0 | a'_p-1 | a_p | b_p | b'_p | s''_p
            N = tensordot(N,W,axes=([4,8],[0,3]))

            # io: a_0 | b_0 | b'_0 | a'_0 | a'_p-1 | a_p | b_p | b'_p | s''_p
            #   @ s''_p | a'_p | a'_p-1
            #  => a_0 | b_0 | b'_0 | a'_0 | a_p | b_p | b'_p | a'_p
            N = tensordot(N,self._dagger(M),axes=([4,8],[2,0])) 

            # index order: a_0 | b_0 | a'_0 | a_p-1 | b_p-1 | a'_p-1
            #            @ s_p | a_p-1 | a_p
            #           => a_0 | b_0 | a'_0 | b_p-1 | a'_p-1 | s_p | a_p
            O = tensordot(O,M ,axes=([3],[1]))

            # index order: a_0 | b_0 | a'_0 | b_p-1 | a'_p-1 | s_p | a_p
            #            @ b_p-1 | b_p | s'_p | s_p
            #           => a_0 | b_0 | a'_0 | a'_p-1 | a_p | b_p | s'_p
            O = tensordot(O,W,axes=([3,5],[0,3]))

            # index order: a_0 | b_0 | a'_0 | a'_p-1 | a_p | b_p | s'_p
            #            @ s'_p | a'_p | a'_p-1
            #           => a_0 | b_0 | a'_0 | a_p | b_p | a'_p
            O = tensordot(O,self._dagger(M),axes=([3,6],[2,0]))

        # N index order:  a_0 | b_0 | b'_0 | a'_0  |  a_L | b_L | b'_L | a'_L
        #             => (a_0 | b_0 | b'_0 | a'_0) | (a_L | b_L | b'_L | a'_L)
        # O index order:  a_0 | b_0 | a'_0  |  a_L | b_L | a'_L
        #             => (a_0 | b_0 | a'_0) | (a_L | b_L | a'_L)
        if self._useReducedTensors:

            rt.combine_axes_for_reduced_tensor(N,1,4,4)
            rt.combine_axes_for_reduced_tensor(N,1,0,4)

            rt.combine_axes_for_reduced_tensor(O,1,3,3)
            rt.combine_axes_for_reduced_tensor(O,1,0,3)

        else:

            N = N.reshape(N.shape[0]*N.shape[1]*N.shape[2]*N.shape[3],
                          N.shape[4]*N.shape[5]*N.shape[6]*N.shape[7])

            O = O.reshape(O.shape[0]*O.shape[1]*O.shape[2],
                          O.shape[3]*O.shape[4]*O.shape[5])

        # calculate trace
        N = trace(N)
        O = trace(O)

        # return results
        if return_all:
            return np.real(N - O**2), np.real(N), np.real(O)
        else:
            return np.real(N - O**2), np.real(O)

    def _calc_next_L_tensor(self,num):
        '''
        Calculates the next 'num' L-tensor. Last L-tensor is the whole
        contraction of the MPS-MPO-MPS network. If the function is then called
        again, it raises a ValueError since a next L-tensor does not exist.
        Also raises an error if the false LR driver is used or periodic
        boundary conditions are used.
        '''

        if self.get_boundary_condition() != 'OBC':
            raise TypeError('L/R tensors cannot be used if open boundary '
                            'conditions are not used.')

        for _ in range(num):

            site = len(self._L_tensors) - 1

            if site >= self.get_length():
                raise ValueError('Cannot calculate next L-tensor. '
                                 'All L-tensors are already calculated.')
            elif site == -1:

                # If zero L-tensors exist, create dummy tensor

                # index order: a_0 | b_0 | a'_0
                if self._useReducedTensors:
                    q_L0 = self._MPS[0].q_vectors[1][0]
                    q_L1 = self._MPO[0].q_vectors[0][0]
                    self._L_tensors = [rt.reducedTensor({(q_L0,q_L1,q_L0):
                                                         np.array([[[1.0]]])},
                                                     [(q_L0,),(q_L1,),(q_L0,)],
                                                        [1,1,-1],
                                                        sectors_given=True)]
                else:
                    self._L_tensors = [np.array([[[1.0]]])]
                continue

            # perform contraction

            # index order: a_p-1 | b_p-1 | a'_p-1 @ s_p | a_p-1 | a_p
            #           => b_p-1 | a'_p-1 | s_p | a_p
            L = tensordot(self._L_tensors[-1],self.get_site(site),
                          axes=([0],[1]))

            # io: b_p-1 | a'_p-1 | s_p | a_p @ b_p-1 | b_p | s'_p | s_p
            #  => a'_p-1 | a_p | b_p | s'_p
            L = tensordot(L,self._MPO[site],axes=([0,2],[0,3]))

            # index order: a'_p-1 | a_p | b_p | s'_p @ s'_p | a'_p | a'_p-1
            #           => a_p | b_p | a'_p
            L = tensordot(L,self._dagger(self.get_site(site)),
                          axes=([0,3],[2,0]))

            # set new L-tensor
            self._L_tensors.append(L)

    def _calc_next_R_tensor(self,num):
        '''
        Calculates the next 'num' R-tensors. Last R-tensor is the whole
        contraction of the MPS-MPO-MPS network. If the function is then called
        again, it a raises a ValueError since a next R-tensor does not exist.
        Also raises an error if the false LR driver is used or periodic
        boundary conditions are used.
        '''

        if self.get_boundary_condition() != 'OBC':
            raise TypeError('L/R tensors cannot be used if open boundary '
                            'conditions are not used.')

        for _ in range(num):

            site = self.get_length() - len(self._R_tensors)

            if site < 0:
                raise ValueError('Cannot calculate next R-tensor. '
                                 'All R-tensors are already calculated.')
            elif site == self.get_length():

                # If zero L-tensors exist, create dummy tensor

                # index order: a_L | b_L | a'_L
                if self._useReducedTensors:
                    q_R0 = self._MPS[-1].q_vectors[2][0]
                    q_R1 = self._MPO[-1].q_vectors[1][0]
                    self._R_tensors = [rt.reducedTensor({(q_R0,q_R1,q_R0):
                                                         np.array([[[1.0]]])},
                                                     [(q_R0,),(q_R1,),(q_R0,)],
                                                        [-1,-1,1],
                                                        sectors_given=True)]
                else:
                    self._R_tensors = [np.array([[[1.0]]])]
                continue

            # perform contraction

            # index order: a_p | b_p | a'_p @ s_p | a_p-1 | a_p
            #           => b_p | a'_p | s_p | a_p-1
            R = tensordot(self._R_tensors[-1],self.get_site(site),
                          axes=([0],[2]))

            # index order: b_p | a'_p | s_p | a_p-1 @ b_p-1 | b_p | s'_p | s_p
            #           => a'_p | a_p-1 | b_p-1 | s'_p
            R = tensordot(R,self._MPO[site],axes=([0,2],[1,3]))

            # index order: a'_p | a_p-1 | b_p-1 | s'_p @ s'_p | a'_p | a'_p-1
            #           => a_p-1 | b_p-1 | a'_p-1
            R = tensordot(R,self._dagger(self.get_site(site)),
                          axes=([0,3],[1,0]))

            # set new R-tensor
            self._R_tensors.append(R)

    def _recalculate_L_tensors_behind_pos(self,pos):
        '''
        Recalculates all L tensors behind 'pos', the first one recalculated
        therefore being L['pos'+1] (for positive 'pos').
        '''

        # make 'pos' positive
        if pos < 0:
            pos = pos + self.get_length()

        # select behavior based on LR_driver
        if self.get_LR_driver() == 'none':
            return
        elif self.get_LR_driver() == 'complete':
            L_border = self.get_length()
        elif self.get_LR_driver() == 'manual':
            L_border = self._L_border
        elif self.get_LR_driver() == 'locked':
            L_border = self._A_border

        # delete old L tensors
        del self._L_tensors[pos + 1:]

        # make 'L_border' positive (+1 since there are L+1 possible positions)
        if L_border < 0:
            L_border = self._R_border + self.get_length() + 1

        # recalculate L tensors
        self._calc_next_L_tensor(L_border - pos)

    def _recalculate_R_tensors_before_pos(self,pos):
        '''
        Recalculates all L tensors behind 'pos', the first one recalculated
        therefore being L['pos'+1] (for positive 'pos').
        '''

        # make 'pos' positive
        if pos < 0:
            pos = pos + self.get_length() + 1

        # select behavior based on LR_driver
        if self.get_LR_driver() == 'none':
            return
        elif self.get_LR_driver() == 'complete':
            R_border = 0
        elif self.get_LR_driver() == 'manual':
            R_border = self._R_border
        elif self.get_LR_driver() == 'locked':
            R_border = self._B_border

        # delete old R tensors
        del self._R_tensors[self.get_length() - pos - 1 + 1 + 1:]

        # make 'R_border' positive (+1 since there are L+1 possible positions)
        if R_border < 0:
            R_border = R_border + self.get_length() + 1

        # recalculate L tensors
        self._calc_next_R_tensor(pos - R_border)

    def set_L_tensor_border(self,pos):
        '''
        For the manual LR driver. Set the border of the L tensors to be at
        position 'pos'. Negative indices are allowed to count from the right
        end of the spin chain. Because of this 'pos' may lie in the interval
        [-L-1,L] if L is the length of the spin chain. If 'pos' is positive,
        the last L tensor incorporates the site tensor at position 'pos'.
        '''

        # check whether the correct driver is used.
        if self.get_LR_driver() != 'manual':
            raise ValueError("The L border is only defined for the LR driver "
                             "'manual' and can therefore only be used with "
                             "this driver. The driver currently used however "
                             "is '{}'.".format(self.get_LR_driver()))

        # check if position is valid, may lie in [-L-1,L].
        if pos < -self.get_length()-1 or pos > self.get_length():
            raise ValueError('pos must lie in the interval [-L-1,L]. '
                             '{} is an invalid value.'.format(pos))

        # determine old and new L tensor border using positive indices
        old_pos = (self._L_border + self.get_length() + 1
                   if self._L_border < 0 else self._L_border)
        new_pos = pos + self.get_length() + 1 if pos < 0 else pos

        self._L_border = pos

        # delete or add L tensors as needed
        if old_pos == new_pos:
            # nothing changes
            return
        if old_pos > new_pos:
            # no new tensors, delete old tensors
            delete = old_pos - new_pos
            del self._L_tensors[-delete:]
        else:
            # calculate new tensors
            add = new_pos - old_pos
            self._calc_next_L_tensor(add)

    def set_R_tensor_border(self,pos):
        '''
        For the manual LR driver. Set the border of the R tensors to be at
        position 'pos'. Negative indices are allowed to count from the right
        end of the spin chain. Because of this 'pos' may lie in the interval
        [-L-1,L] if L is the length of the spin chain. If 'pos' is positive,
        the last R tensor incorporates the site tensor at position 'pos+1'.
        '''

        # check whether the correct driver is used.
        if self.get_LR_driver() != 'manual':
            raise ValueError("The R border is only defined for the LR driver "
                             "'manual' and can therefore only be used with "
                             "this driver. The driver currently used however "
                             "is '{}'.".format(self.get_LR_driver()))
            
        # check if position is valid, may lie in [-L-1,L].
        if pos < -self.get_length()-1 or pos > self.get_length():
            raise ValueError('pos must lie in the interval [-L-1,L]. '
                             '{} is an invalid value.'.format(pos))

        # determine old and new R tensor border using positive indices
        old_pos = (self._R_border + self.get_length() + 1
                   if self._R_border < 0 else self._R_border)
        new_pos = pos + self.get_length() + 1 if pos < 0 else pos

        self._R_border = new_pos

        # delete or add R tensors as needed
        if old_pos == new_pos:
            # nothing changes
            return
        if old_pos > new_pos:
            # calculate new tensors
            add = old_pos - new_pos
            self._calc_next_R_tensor(add)
        else:
            # no new tensors, delete old tensors
            delete = new_pos - old_pos
            del self._R_tensors[-delete:]

    def get_L_tensor_border(self):
        '''
        Returns the L border. Only possible if the LR driver is set to
        'manual'. If this is not the case, a ValueError will be raised.
        Returned position indicates that all tensors left of bond a_p are
        incorporated into the L tensors with the right-most site tensor being
        the one at the returned position.
        '''

        if self._LR_driver != 'manual':
            raise ValueError("The L border is only defined for the LR driver "
                             "'manual' and can therefore only be used with "
                             "this driver. The driver currently used, however,"
                             " is '{}'.".format(self.get_LR_driver()))

        return self._L_border

    def get_R_tensor_border(self):
        '''
        Returns the R border. Only possible if the LR driver is set to
        'manual'. If this is not the case, a ValueError will be raised.
        Returned position indicates that all tensors right of bond a_p are
        incorporated into the R tensors with the left-most site tensor being
        the one that is one place to the right of the returned position.
        '''

        if self._LR_driver != 'manual':
            raise ValueError("The R border is only defined for the LR driver "
                             "'manual' and can therefore only be used with "
                             "this driver. The driver currently used, however,"
                             " is '{}'.".format(self.get_LR_driver()))

        return self._R_border

    def get_number_L_tensors(self):
        '''
        Returns the number of L tensors. This number will at least be one, even
        if the usage of the LR tensors is deactivated. The first entry is a
        dummy tensor. The maximal number is L+1.
        '''

        return len(self._L_tensors)

    def get_number_R_tensors(self):
        '''
        Returns the number of R tensors. This number will at least be one, even
        if the usage of the LR tensors is deactivated. The first entry is a
        dummy tensor. The maximal number is L+1.
        '''

        return len(self._R_tensors)

    def get_L_tensor(self,pos=-1):
        '''
        Returns the L tensor at position 'pos'. Negative indices may be used.
        They refer to the position in the chain of L tensors and not to the
        position in the MPS. 'pos' is preset to -1 giving the 'newest' L
        tensor. Index order of returned tensor: a_p | b_p | a'_p.
        '''

        # check if position is valid.
        if (pos < -self.get_number_L_tensors()
            or pos > self.get_number_L_tensors()-1):
            raise ValueError('Only {} '.format(self.get_number_L_tensors())+
                             'L tensor(s) exist. {} '.format(pos)+
                             'is therefore an invalid choice.')

        return self._L_tensors[pos]

    def get_R_tensor(self,pos=-1):
        '''
        Returns the R tensor at position 'pos'. Negative indices may be used.
        They refer to the position in the chain of R tensors and not to the
        position in the MPS. 'pos' is preset to -1 giving the 'newest' R
        tensor. Index order of returned tensor: a_p | b_p | a'_p.
        '''

        # check if position is valid.
        if (pos < -self.get_number_R_tensors() or
            pos > self.get_number_R_tensors()-1):
            raise ValueError('Only {} '.format(self.get_number_R_tensors())+
                             'R tensor(s) exist. {} '.format(pos)+
                             'is therefore an invalid choice.')

        return self._R_tensors[pos]

    #################################################################
    # overwrite functions from class MPS to include L and R tensors #
    #################################################################

    def set_site(self,pos,site_tensor,different_shape=False):
        '''
        Change a site tensor at a particular position to the supplied site
        tensor. The new site tensor will be treated as being not normalized.
        This affects the position of the A- and B-boundaries. Furthermore, L
        and R tensors are recalculated as needed. If the tensor is meant to
        have a different shape or different charge vectors if reduced tensors
        are used, it must be explicitly mentioned by setting
        different_shape=True to deactivate compatibility tests. By changing the
        shape or the charge vector structure, the MPS becomes illegal. The user
        has to ensure that further operations on the MPS makes the MPS legal
        again. The index order of the tensor 'site_tensor'
        must be: s_p | a_p-1 | a_p.
        '''

        # call parent function
        x = super().set_site(pos,site_tensor,different_shape)

        # recalculate L/R tensors as needed
        self._recalculate_L_tensors_behind_pos(pos)
        self._recalculate_R_tensors_before_pos(pos+1)

        # forward any output from parent function
        return x

    def set_twosite_tensor(self,pos_left,two_site_tensor,normalization,D=None,
                           max_truncation_error=None,return_USV=False):

        # call parent function
        x = super().set_twosite_tensor(pos_left,two_site_tensor,normalization,
                                       D,max_truncation_error,return_USV)

        # recalculate L/R tensors as needed.
        self._recalculate_L_tensors_behind_pos(pos_left)
        self._recalculate_R_tensors_before_pos(pos_left+2)

        # forward any output from parent function
        return x

    def set_several_site_tensors(self,pos_start,list_of_tensors):

        # call parent function
        x = super().set_several_site_tensors(pos_start,list_of_tensors)

        # recalculate L/R tensors as needed
        self._recalculate_L_tensors_behind_pos(pos_start)
        self._recalculate_R_tensors_before_pos(
            pos_start+len(list_of_tensors)-1)

        # forward any output from parent function
        return x

    def move_left_normalized_boundary_to_the_right(self,steps=1):
        
        # call parent function
        x = super().move_left_normalized_boundary_to_the_right(steps)

        # recalculate L/R tensors as needed
        old_L_border = self.get_left_normalized_border()
        self._recalculate_L_tensors_behind_pos(old_L_border)
        self._recalculate_R_tensors_before_pos(
            self.get_left_normalized_border())

        # forward any output from parent function
        return x

    def move_right_normalized_boundary_to_the_left(self,steps=1):

        # call parent function
        x = super().move_right_normalized_boundary_to_the_left(steps)

        # recalculate L/R tensors as needed
        old_R_border = self.get_right_normalized_border()
        self._recalculate_R_tensors_before_pos(old_R_border)
        self._recalculate_L_tensors_behind_pos(
            self.get_right_normalized_border())

        # forward any output from parent function
        return x

    def insert_site_tensor(self,pos,site_tensor,MPO_site,
                           different_shape=False):

        # insert MPO site
        if self._MPO is None:
            self._MPO = [MPO_site]
        else:
            self._MPO.insert(pos,MPO_site)

        # call parent function
        x = super().insert_site_tensor(pos,site_tensor,different_shape)

        # recalculate L/R tensors as needed
        self._R_border += 1
        self._recalculate_L_tensors_behind_pos(pos)
        self._recalculate_R_tensors_before_pos(pos+1)

        # forward any output from parent function
        return x

    def insert_twosite_tensor(self,pos,two_site_tensor,MPO_site_left,
                              MPO_site_right,normalization,D=None,
                              max_truncation_error=None,return_USV=False):

        # insert MPO sites
        if self._MPO is None:
            self._MPO = [MPO_site_left,MPO_site_right]
        else:
            self._MPO.insert(pos,MPO_site_right)
            self._MPO.insert(pos,MPO_site_left)

        # call parent function
        x = super().insert_twosite_tensor(pos,two_site_tensor,normalization,D,
                                          max_truncation_error,return_USV)

        # recalculate L/R tensors as needed
        self._R_border += 2
        self._recalculate_L_tensors_behind_pos(pos)
        self._recalculate_R_tensors_before_pos(pos+2)

        # forward any output from parent function
        return x

    def add(self,MPS_system,factor=1.,ignore_multiplicative_factor=True):

        # call parent function
        x = super().add(MPS_system,factor,ignore_multiplicative_factor)

        # recalculate L/R tensors as needed
        self._recalculate_L_tensors_behind_pos(0)
        self._recalculate_R_tensors_before_pos(-1)

        # forward any output from parent function
        return x

    def enlarge(self,D):
        '''
        Enlarges the MPS by padding it with zeros to the new local bond
        dimension. If the new local bond dimension is smaller than the current
        one, the MPS will not be compressed but individual bonds which are
        smaller will be increased. If the MPS has open boundary conditions, the
        exponential decay toward the edges is ensured. Also enlarges the
        L/R tensors. Cannot be used for reduced tensors.
        '''

        # call parent function
        x = super().enlarge(D)

        # enlarge L tensors
        for pos in range(len(self._L_tensors)):

            padD=max(0,self._MPS[pos].shape[1]-self._L_tensors[pos].shape[0])
            self._L_tensors[pos] = np.pad(self._L_tensors[pos],
                                          ((0,padD),(0,0),(0,padD)),'constant',
                                          constant_values=0)

        # enlarge R tensors
        for pos in range(len(self._R_tensors)):

            padD=max(0,self._MPS[-pos].shape[1]-self._R_tensors[pos].shape[0])
            self._R_tensors[pos] = np.pad(self._R_tensors[pos],
                                          ((0,padD),(0,0),(0,padD)),'constant',
                                          constant_values=0)

        # forward any output from parent function
        return x
