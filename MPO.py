import numpy as np
import reducedTensor as rt
import copy

from MPS import perform_np_SVD
from MPS import tensordot
from MPS import trace


_assert_parameter_length = ('{} must be an iterable object '
                            'with length {} or a number.')

##########################################################
# internal functions not meant to be called from outside #
##########################################################

def _make_iterable(a,length):
    '''
    If the given object 'a' is not iterable, creates an instance of np.ndarray
    of length 'length' and attempts to set all entries to 'a'. This array is
    then returned. If 'a' is iterable, returns 'a' without any modifications.
    '''

    if not (type(a) is list or type(a) is tuple or type(a) is np.ndarray):
        return a * np.ones(length)
    else:
        return a

def _get_local_bond_dimensions_Uform(L_row,L_col,IR,IC,IB):
    '''
    Under the assumption that an MPO describes a two-dimensional spin lattice
    by using the U method (shown below), calculate the local bond dimensions
    between the individual MPO site tensors based on the number of rows
    'L_row', columns 'L_col' in the spin lattice in addition to the number of
    interactions happening only for rows 'IR', only for columns 'IC' or for
    both 'IB'. Includes the trivial local bond dimensions at the border. It is
    advisable to have L_row <= L_col.

    U-method:

    o   o---o   o---o
    |   |   |   |   |
    o   o   o   o   o
    |   |   |   |   |
    o   o   o   o   o
    |   |   |   |   |
    o---o   o---o   o

    '''

    Dw = [1]

    for C in range(L_col):
        for R in range(L_row):

            if C == L_col -1 and R == L_row -1:
                break

            if C == L_col -1:
                Dw.append(2*(IB+IC)*L_row - (R+1+L_row)*(IB+IC) + IR + 2)
            else:
                Dw.append(2*(IB+IC)*L_row - (R+1)*(IB+IC)       + IR + 2)

                # length of long range interaction
                # + short-range only interactions + 2

    Dw.append(1)

    return Dw

def _find_out_type(site):
    '''
    Try to guess the type of the given matrix of shape (2x2). Returns the
    following values:
    ''  : zero matrix
    '1' : identity matrix
    '2' : multiple of identity matrix
    'X' : Pauli x matrix
    'x' : multiple of Pauli x matrix
    'Y' : Pauli y matrix
    'y' : multiple of Pauli y matrix
    'Z' : Pauli z matrix
    'z' : multiple of Pauli z matrix
    'A' : matrix of structure: multiple of Pauli x  + multiple of Pauli z
    'D' : diagonal matrix
    '?' : unknown matrix structure
    '''

    if np.linalg.norm(site) == 0:
        return ''
    elif np.linalg.norm(site - np.eye(2)) == 0:
        return '1'
    elif site[0,0] == site[1,1] and site[0,1] == 0 and site[1,0] == 0:
        return '2'
    elif site[0,1] ==  site[1,0] and site[0,0] == 0 and site[1,1] == 0:
        if site[0,1] == 1.:
            return 'X'
        else:
            return 'x'
    elif site[0,1] == -site[1,0] and site[0,0] == 0 and site[1,1] == 0:
        if site[0,1] == -1.j:
            return 'Y'
        else:
            return 'y'
    elif site[0,0] == -site[1,1] and site[0,1] == 0 and site[1,0] == 0:
        if site[0,0] == 1.:
            return 'Z'
        else:
            return 'z'
    elif site[0,0] == -site[1,1] and site[0,1] == site[1,0]:
        # z+x
        return 'A'
    elif site[0,1] == 0 and site[1,0] == 0:
        return 'D'
    elif site[0,0] == 0 and site[0,1] == 0 and site[1,1] == 0:
        return '+'
    elif site[0,0] == 0 and site[1,0] == 0 and site[1,1] == 0:
        return '-'
    else:
        return '?'

####################################################
# functions performing analytical analyses on MPOs #
####################################################
    
def turnMPOanalytic(M):
    '''
    Analyses a given MPO 'M' and attempts to find out the internal structure of
    it. Returns a list containing lists of lists which represent the site
    tensors of the given MPS but where the physical (d x d) matrices of the MPO
    have been replaced by strings representing the structure of the matrix at
    the corresponding position in the corresponding site tensor of 'M'.
    '''

    N = []
    for i,site in enumerate(M):
        N2 = []

        for col in range(len(site)):
            N2.append([])
            for row in range(len(site[0])):
                string = _find_out_type(site[col,row])
                if string != '' and string != '1':
                    string += '_{}'.format(i+1)

                N2[-1].append(string)

        N.append(N2)

    return N

def multiply_analytic_matrices(S1,S2):
    '''
    Multiplies two analytical matrices as returned by the function
    'turnMPOanalytic'. Returns a new analytic matrix representing the matrix
    product of 'S1' and 'S2'.
    '''

    I = len(S1)
    K = len(S1[0])
    J = len(S2[0])

    S3 = [['' for i in range(J)] for i in range(I)]

    for i in range(I):
        for j in range(J):
            for k in range(K):

                # calc result
                if S1[i][k] == ''  or  S2[k][j] == '':
                    continue
                elif S1[i][k] == '1' and S2[k][j] != '1':
                    r = S2[k][j]
                elif S1[i][k] != '1' and S2[k][j] == '1':
                    r = S1[i][k]
                elif S1[i][k] == '1' and S2[k][j] == '1':
                    r = '1'
                else:
                    if '+' in S1[i][k] and '+' in S2[k][j]:
                        r = '('+S1[i][k]+')('+S2[k][j]+')'
                    elif '+' in S1[i][k] and '+' not in S2[k][j]:
                        r = '('+S1[i][k]+')'+S2[k][j]
                    elif '+' not in S1[i][k] and '+' in S2[k][j]:
                        r = S1[i][k]+'('+S2[k][j]+')'
                    else:
                        r = S1[i][k]+S2[k][j]

                if S3[i][j] == '':
                    S3[i][j] = r
                else:
                    S3[i][j] += '+' + r

    return S3

def contract_analytic_MPO(S):
    '''
    Contracts an entire analytic MPO. Gives a string describing the Hamiltonian
    which is encoded in the MPO 'S'.
    '''

    A = multiply_analytic_matrices(S[0],S[1])
    for i in range(2,len(S)):

        A = multiply_analytic_matrices(A,S[i])

    return A[-1][0]

############################
# functions involving MPOs #
############################


def contract_MPO(MPO_):
    '''
    Takes an MPO and contracts it into a high-dimensional tensor which, in
    turn, is reshaped into a matrix after its dimensions got sorted
    appropriately. The sub-dimensional structure of the matrix is as follows:
    ((s1,s2,...,sn),(s'1,s'2,...,s'n)). The matrix is of shape (2**L,2**L).
    '''

    L = len(MPO_)

    # contract MPO
    O = np.tensordot(MPO_[0],MPO_[1],axes=((-3),(0)))
    for i in range(2,L):
        O = np.tensordot(O,MPO_[i],axes=((-3),(0)))

    # remove trivial axes
    O = O.reshape(*2*L*[2])

    # sort axes
    pos_l = []
    for i in range(1,L+1):
        pos_l.extend([i,-i])

    # first half
    for i in range(1,L+1):
        index = pos_l.index(i)
        pos_l[i-1],pos_l[index] = pos_l[index],pos_l[i-1]
        O = np.swapaxes(O,i-1,index)

    # second half
    for i in range(1,L+1):
        index = pos_l.index(-i)
        pos_l[i-1+L],pos_l[index] = pos_l[index],pos_l[i-1+L]
        O = np.swapaxes(O,i-1+L,index)

    # shape to matrix
    O = O.reshape(2**L,2**L)

    return O


def compress_MPO_SVD(MPO_,D=None,max_truncation_error=None,direction='right'):
    '''
    Compress the MPO by employing the singular value decomposition on every
    bond between neighbouring MPO site tensors. This is done by first summing
    over this bond, then performing the SVD, followed by cutting away a number
    of singular values and adjusting the sizes of the involved matrices
    accordingly. Does not work for reduced tensors.

    This function takes a number of arguments:
    MPO_                 : The MPO to be compressed.
    D                    : The local bond dimension to which the sites should
                           be cut down to. Default is None. If D=None, the
                           local bond dimension of the bond being optimised is
                           used.
    max_truncation_error : The largest truncation error to be tolerated.
                           Is overruled by D if necessary. Default is None.
    direction            : Defines the direction toward which to sweep.
                           Default is 'right', sweeping from left to right.

    Returns the [D_l,tr_l,S_l] containing three lists storing the new local
    bond dimension for each bond being optimised, the truncation error
    encountered during the optimisation and the singular values present on each
    bond after the optimisation took place.
    '''

    def optimise_site():
        '''
        Compresses a bond between two sites by calculating the respective MPO
        two-site tensor and performing a singular value decomposition
        afterwards.
        '''

        D_here = D if D is not None else MPO_[l].shape[1]

        Dleft  = MPO_[l].shape[0]
        Dright = MPO_[l+1].shape[1]

        # index order: b_l-1 | b_l | s'_l | s_l @ b_l | b_l+1 | s'_l+1 | s_l+1
        #           => b_l-1 | s'_l | s_l | b_l+1 | s'_l+1 | s_l+1
        M = np.tensordot(MPO_[l],MPO_[l+1],axes=((1),(0)))

        # index order:  b_l-1 | s'_l | s_l  |  b_l+1 | s'_l+1 | s_l+1
        #           => (b_l-1 | s'_l | s_l) | (b_l+1 | s'_l+1 | s_l+1)
        M = M.reshape([M.shape[0]*M.shape[1]*M.shape[2],
                       M.shape[3]*M.shape[4]*M.shape[5]])

        X = perform_np_SVD(M, D=D_here,
                           max_truncation_error=max_truncation_error,
                           normalize='S', return_=return_string)

        D_here, truncation_error, U, V, S = X
        D_l.append(D_here)
        tr_l.append(truncation_error)
        S_l.append(S)

        # index order: (b_l-1 | s'_l | s_l) | b'_l => b_l-1 | s'_l | s_l | b'_l
        U = U.reshape(Dleft,d,d,D_here)

        # index order: b_l-1 | s'_l | s_l | b'_l => b_l-1 | s'_l | b'_l | s_l
        U = U.swapaxes(2,3)

        # index order: b_l-1 | s'_l | b'_l | s_l => b_l-1 | b'_l | s'_l | s_l
        U = U.swapaxes(1,2)

        # index order: b'_l | (b_l+1 | s'_l+1 | s_l+1)
        #           => b'_l |  b_l+1 | s'_l+1 | s_l+1
        V = V.reshape(D_here,Dright,d,d)

        MPO_[l]   = U
        MPO_[l+1] = V

    L = len(MPO_)
    d = 2

    if direction in ['left','right']:

        D_l,tr_l,S_l = [],[],[]

        if direction == 'right':
            range_lim = (0,L-1)
            return_string = 'U|SV|S'
        elif direction == 'left':
            range_lim = (L-2,-1,-1)
            return_string = 'US|V|S'
        else:
            raise ValueError("'direction' must be either 'left' or 'right'.")

        for l in range(*range_lim):
            optimise_site()

    elif direction == 'inwards':

        D_l,tr_l,S_l = [],[],[]

        lastLeft = (L-1)//2

        # left half
        direction = 'right'
        return_string = 'U|SV|S'
        for l in range(lastLeft):
            optimise_site()

        # right half
        direction = 'right'
        return_string = 'US|V|S'
        for l in range(L-2,lastLeft-1,-1):
            optimise_site()

    elif direction == 'outwards':

        D_l,tr_l,S_l = [],[],[]

        start = L//2

        # left half
        direction = 'left'
        return_string = 'US|V|S'
        for l in range(start-1,-1,-1):
            optimise_site()

        # right half
        direction = 'right'
        return_string = 'U|SV|S'
        for l in range(start,L-1):
            optimise_site()

    return D_l,tr_l,S_l

def MPO_overlap(MPO1,MPO2):
    '''
    Calculate the overlap between two MPOs. This is achieved by treating both
    physical indices as one thus interpreting the MPOs as MPSs with squared
    local state space dimensions.
    '''

    L = len(MPO1)
    if len(MPO2) != L:
        raise ValueError('The two MPOs are of unequal length.')

    # index order: b_0 | b_1 | s'_1 | s_1 @ b'_0 | b'_1 | s_1 | s'_1
    #           => b_0 | b_1 | b'_0 | b'_1
    overlap = tensordot(np.conjugate(MPO1[0]),MPO2[0],axes=([2,3],[3,2]))

    # index order: b_0 | b_1 | b'_0 | b'_1 => b_0 | b'_0 | b_1 | b'_1
    overlap = overlap.swapaxes(1,2)

    for i in range(1,L):

        # index order: b_0 | b'_0 | b_p-1 | b'_p-1 @ b_p-1 | b_p | s'_p | s_p
        #           => b_0 | b'_0 | b'_p-1 | b_p | s'_p | s_p
        overlap = tensordot(overlap,np.conjugate(MPO1[i]),axes=([2],[0]))

        # index order: b_0 | b'_0 | b'_p-1 | b_p | s'_p | s_p
        #            @ b'_p-1 | b'_p | s_p | s'_p
        #           => b_0 | b'_0 | b_p | b'_p
        overlap = tensordot(overlap,MPO2[i],axes=([2,4,5],[0,3,2]))

    # index order: b_0 | b'_0 | b_L | b'_L => (b_0 | b'_0) | (b_L | b'_L)
    if type(MPO1[0]) is rt.reducedTensor:
        rt.combine_axes_for_reduced_tensor(overlap,1,2,2)
        rt.combine_axes_for_reduced_tensor(overlap,1,0,2)
    else:
        overlap = overlap.reshape(overlap.shape[0]*overlap.shape[1],
                                  overlap.shape[2]*overlap.shape[3])

    return trace(overlap)


def MPO_norm(MPO):
    '''
    Retuns the norm of the MPO.
    '''

    return np.sqrt(np.abs(MPO_overlap(MPO,MPO)))


def MPO_normalize(MPO):
    '''
    Normalises the given MPO in-place. Does not work for reduced tensors or
    periodic boundary conditions.
    '''

    compress_MPO_SVD(MPO)
    norm = MPO_norm(MPO)

    MPO[-1] /= norm


def get_MPO_local_bond_dimension(MPO):
    '''
    Returns the local bond dimensions of the MPO for the individual bonds.
    The trivial bonds at the ends of the MPO are omitted.
    '''

    return [MPO[i].shape[1] for i in range(len(MPO)-1)]


def add_MPO_together(list_of_MPOs):
    '''
    Takes a list of MPOs that is added together into a new MPO. This MPO is
    then returned. The size of the bonds are the sum of the sizes of every MPO
    in the list. Compression is not performed in this function. To utilize
    compression during the summing process see: add_MPO_together_compress. Does
    not work with reduced tensors.
    '''

    N = len(list_of_MPOs)
    L = len(list_of_MPOs[0])
    d = 2

    # check that all MPOs have the same length
    for site in list_of_MPOs:
        if len(site) != L:
            raise ValueError('Not all MPOs are of the same length.')

    # perform the addition process
    M = []

    # loop over MPOs
    for l in range(L):
        VDL = [list_of_MPOs[i][l].shape[0] for i in range(N)] # virt.dim. left
        VDR = [list_of_MPOs[i][l].shape[1] for i in range(N)] # virt.dim. right

        if l == 0:
            A = np.zeros([1,np.sum(VDR),d,d],dtype=np.complex)
        elif l == L-1:
            A = np.zeros([np.sum(VDL),1,d,d],dtype=np.complex)
        else:
            A = np.zeros([np.sum(VDL),np.sum(VDR),d,d],dtype=np.complex)

        CPL,CPR = 0,0 # current position left/right

        # loop over MPO site tensors
        for n in range(N):
            A[CPL:CPL+VDL[n],CPR:CPR+VDR[n]] = list_of_MPOs[n][l]
            CPL = CPL + VDL[n] if l >  0  else 0
            CPR = CPR + VDR[n] if l < L-1 else 0

        M.append(A)

    return M


def add_MPO_together_compress(list_of_MPOs,LBD_max=100,D=None,
                              max_truncation_error=1e-14,print_progress=False):
    '''
    Add all MPOs together given in the list_of_MPOs and, while doing so,
    compress the resulting MPO in the process. This function takes the
    following arguments:
    list_of_MPOs         : A list containing the MPOs that should be added
                           together.
    LBD_max              : During the adding process of the MPOs, a compression
                           is triggered whenever the local bond dimension of
                           the MPO representing the sum reaches this number.
                           Default is 100.
    D                    : The local bond dimension to which the MPO is
                           compressed when a compression is triggered.
                           Default is None meaning that the local bond
                           dimension of the MPO to be compressed is used.
    max_truncation_error : The truncation error that is tolerated during
                           compressions. May be overruled by D if necessary.
                           Default is 1e-14.
    print_progress       : Prints the progress of the calculation to the
                           standard output. Default is False.

    Returns an MPO that is the sum of all supplied MPOs.
    '''

    LBD = 0 # local bond dimension

    add_l = []
    for i in range(len(list_of_MPOs)):

        add_l.append(list_of_MPOs[i])
        LBD += np.max(get_MPO_local_bond_dimension(list_of_MPOs[i]))

        if LBD > LBD_max or i == len(list_of_MPOs)-1:

            if print_progress:
                print('\rcompression @ {}/{}'.format(i+1,len(list_of_MPOs)),
                      end='')

            MPO_ = add_MPO_together(add_l)
            _ = compress_MPO_SVD(MPO_,D=D,
                                 max_truncation_error=max_truncation_error,
                                 direction='right')

            _ = compress_MPO_SVD(MPO_,D=D,
                                 max_truncation_error=max_truncation_error,
                                 direction='left' )

            add_l = [MPO_]
            LBD = 0

    return add_l[0]

def print_MPO_site_structure(site):
    '''
    Prints the structure of an MPO site tensor which is of shape (Dw,Dw,d,d)
    where we interpret the site as a matrix of shape (Dw,Dw) whose entries
    are matrices of shape (d,d). This function prints the site matrix and
    prints an 'x' if the submatrix at that position is non-zero. Does not work
    for reduced tensors.
    '''

    print('  +'+2*site.shape[1]*'-'+'+')
    for i in range(site.shape[0]):

        print('{:2d}'.format(i+1)+'|',end='')
        for j in range(site.shape[1]):

            if np.linalg.norm(site[i,j]) > 0:
                print(' '+_find_out_type(site[i,j]),end='')
            else:
                print('  ',end='')
        print('|')
    print('  +'+2*site.shape[1]*'-'+'+')

    print('   ',end='')
    for i in range(site.shape[1]):
        print('{:2d}'.format(i+1),end='')

    print()

###############################
# functions that provide MPOs #
###############################

def single_interaction_MPO(string,prefactors):
    '''
    Creates an MPO for a single interaction term. The interaction is given by
    the string 'string', which is of the form 'xyzxyz' and may contain the
    characters 'x', 'y', 'z', '1' and '0' representing the respective Pauli
    matrices, the identity matrix and the zero matrix. Using a zero matrix is
    allowed but causes the entire MPO to encode a zero matrix. The length of
    the MPO is given indirectly by the string and prefactors as
    len(prefactors) + len(string) - 1. The returned MPO respects no symmetries.
    '''

    # The dictionary detailing the allowed matrices
    s = {'x':np.array([[0,  1 ],[1 ,  0]]),
         'y':np.array([[0, -1j],[1j,  0]]),
         'z':np.array([[1,  0 ],[0 , -1]]),
         '1':np.array([[1,  0 ],[0 ,  1]]),
         '0':np.array([[0,  0 ],[0 ,  0]])}

    L = len(prefactors) + len(string) - 1
    N = len(string)
    M = []

    # site 1
    A = np.zeros([1,N+1,2,2],dtype=np.complex)
    A[0,-1] = s['1']
    A[0,-2] = prefactors[0] * s[string[0]]
    M.append(A)

    # sites in the middle
    for site in range(1,L-1):
        A = np.zeros([N+1,N+1,2,2],dtype=np.complex)
        A[0,0]  = s['1']
        A[-1,-1] = s['1']
        for col in range(N-1):
            if L - site > col:
                A[col+1,col] = s[string[N - col - 1]]
        if site < len(prefactors):
            A[-1,-2] = prefactors[site] * s[string[0]]
        M.append(A)

    # site L
    A = np.zeros([N+1,1,2,2],dtype=np.complex)
    A[0,0] = s['1']
    if len(string) == 1:
        A[1,0] = prefactors[L-1] * s[string[-1]]
    else:
        A[1,0] = s[string[-1]]
    M.append(A)

    return M


def XZZ_Heisenberg_chain(L,J=1.,Delta=1.,h=0.):
    '''
    Returns the Hamiltonian for the XZZ Heisenberg chain for a given length L
    and given parameters J, Delta and h. The returned MPO contains no
    symmetries. If the arguments J, Delta and h are given as numbers, the J_k,
    Delta_k and h_k for each site are set to this value. They can also be given
    as a list in which case they contain the values for each site. Open
    boundary conditions are assumed.

    H =  \sum_{k=1}^{L-1} J_k (Delta_k x_k x_{k+1} + y_k y_{k+1} + z_k z_{k+1})
        -\sum_{k=1}^{L} h_k z_k
    '''

    Sx = np.array([[0,0.5],[0.5,0]])
    Sw = np.array([[0,-0.5],[0.5,0]])
    Sz = np.array([[0.5,0],[0,-0.5]])
    Sp = np.array([[0.,1],[0,0]])
    Sm = np.array([[0.,0],[1,0]])
    I  = np.eye(2)
    O  = np.zeros([2,2])

    H = []

    J     = _make_iterable(  J  ,L-1)
    Delta = _make_iterable(Delta,L-1)
    h     = _make_iterable(  h  ,L  )

    assert len(J)     == L-1, _assert_parameter_length.format(  'J'  ,L-1)
    assert len(Delta) == L-1, _assert_parameter_length.format('Delta',L-1)
    assert len(h)     == L  , _assert_parameter_length.format(  'h'  ,L  )

    # first site tensor
    H.append(np.array([[-h[0]*Sz,J[0]*Delta[0]*Sx,J[0]*Sw,J[0]*Sz,I]]))

    for i in range(1,len(h)-1):

        # bulk site tensor
        H.append(np.array([[I , O, O, O, O],
                           [Sx, O, O, O, O],
                           [Sw, O, O, O, O],
                           [Sz, O, O, O, O],
                           [-h[i]*Sz,J[i]*Delta[i]*Sx,J[i]*Sw,J[i]*Sz,I]]))

    # last site tensor
    H.append(np.array([[I],[Sx],[Sw],[Sz],[-h[-1]*Sz]]))

    return H


def XXZ_Heisenberg_chain(L,J=1.,Jz=1.,h=0.):
    '''
    Returns the Hamiltonian for the XXZ Heisenberg chain for a given length L
    and given parameters J, Jz and h. The returned MPO contains no symmetries.
    If the arguments J, Jz and h are given as numbers, the J_k, Jz_k and h_k
    for each site are set to this value. They can also be given as a list in
    which case they contain the values for each site. Open boundary conditions
    are assumed.

    H = -\sum_{k=1}^{L-1} J_k (+_k -_{k+1} + -_k +_{k+1}) + Jz_k z_k z_{k+1}
        -\sum_{k=1}^{L} h_k z_k
    '''

    Sz = np.array([[0.5,0],[0,-0.5]])
    Sp = np.array([[0.,1],[0,0]])
    Sm = np.array([[0.,0],[1,0]])
    I  = np.eye(2)
    O  = np.zeros([2,2])

    H = []

    J  = _make_iterable( J,L-1)
    Jz = _make_iterable(Jz,L-1)
    h  = _make_iterable( h,L  )

    assert len(J)  == L-1, _assert_parameter_length.format('J' ,L-1)
    assert len(Jz) == L-1, _assert_parameter_length.format('Jz',L-1)
    assert len(h)  == L  , _assert_parameter_length.format('h' ,L  )

    # first site tensor
    H.append(np.array([[-h[0]*Sz,-J[0]*Sm/2.,-J[0]*Sp/2.,-Jz[0]*Sz,I]]))

    for i in range(1,len(h)-1):

        # bulk site tensor
        H.append(np.array([[I , O, O, O, O],
                           [Sp, O, O, O, O],
                           [Sm, O, O, O, O],
                           [Sz, O, O, O, O],
                           [-h[i]*Sz,-J[i]*Sm/2.,-J[i]*Sp/2.,-Jz[i]*Sz,I]]))

    # last site tensor
    H.append(np.array([[I],[Sp],[Sm],[Sz],[-h[-1]*Sz]]))

    return H


def XXZ_Heisenberg_chain_reduced_MPO(L,J=1.,Jz=1.,h=0.):
    '''
    Returns the Hamiltonian for the XXZ Heisenberg chain for a given length L
    and given parameters J, Jz and h. The returned MPO respects the present
    U(1) symmetry and its site tensors are of type rt.reducedTensor. If the
    arguments J, Jz and h are given as numbers, the J_k, Jz_k and h_k for each
    site are set to this value. They can also be given as a list in which case
    they contain the values for each site. Open boundary conditions are
    assumed.

    H = -\sum_{k=1}^{L-1} J_k (+_k -_{k+1} + -_k +_{k+1}) + Jz_k z_k z_{k+1}
        -\sum_{k=1}^{L} h_k z_k

    '''

    MPO = XXZ_Heisenberg_chain(L=L,J=J,Jz=Jz,h=h)

    q_b0 = [0]
    q_b  = [0 ,-2, 2, 0, 0]
    q_a  = [-1, 1]

    # first site tensor
    MPO[0] = rt.reducedTensor(MPO[0], [q_b0,q_b,q_a,q_a],[-1,1,1,-1],Q=0)

    for i in range(len(MPO)-2):

        # bulk site tensor
        MPO[i+1] = rt.reducedTensor(MPO[i+1], [q_b,q_b,q_a,q_a],
                                    [-1,1,1,-1],Q=0)

    # last site tensor
    MPO[-1] = rt.reducedTensor(MPO[-1], [q_b,q_b0,q_a,q_a],[-1,1,1,-1],Q=0)

    return MPO


def XXZ_Heisenberg_2D_lattice(L_row,L_col,J_row,Jz_row,J_col,Jz_col,h,
                              cylinder=False):
    '''
    Returns the Hamiltonian for the two-dimensional XXZ Heisenberg lattice for
    given side lengths L_row and L_col and given parameters J_row, Jz_row,
    J_col, Jz_col and h. It is advised to set L_row <= L_col in order to
    receive MPOs with smaller local bond dimensions. The returning MPO contains
    no symmetries. If the arguments J_row, Jz_row, J_col, Jz_col and h are
    given as numbers, the J_row_k, Jz_row_k, J_col_k, Jz_col_k and h_k for each
    site are set to this value. Open boundary conditions are assumed.

    The parameters can also be given as a list in which case they contain the
    values for each site, respectively, bond. Open boundary conditions are
    assumed. The length of each list of parameters is the following:

    J_row  | (L_row-1) *  L_col
    Jz_row | (L_row-1) *  L_col
    J_col  |  L_row    * (L_col-1)
    Jz_col |  L_row    * (L_col-1)
    h      |  L_row    *  L_col

    The position of the entries in the parameter list corresponds to the order
    in which the snake encounters the corresponding sites and bonds.

    It is possible to choose cylindrical boundary conditions instead of open
    boundary conditions by setting cylinder=True. For cylindrical boundary
    conditions, the boundary conditions in L_row direction (the short
    direction) turn periodic while the boundary conditions in L_col direction
    (the long direction) remain open. The additional values needed for J_row
    and Jz_row are set to 1 so that the interpretations of the numbers stored
    in J_row and Jz_row remain the same.

    H = -\sum_{<k,l>} J (+_k -_l + -_k +_l) + Jz z_k z_l
        -\sum_{k=1}^{L} h_k z_k

    Form of the snake:

    o   o   o   o   o
    |  /|  /|  /|  /|
    o / o / o / o / o
    |/  |/  |/  |/  |
    o   o   o   o   o

    '''

    # set 2x2 matrices
    Sx = 0.5*np.array([[0,1],[1,0]])
    Sy = 0.5*np.array([[0,-1j],[1j,0]])
    Sz = 0.5*np.array([[1,0],[0,-1]])
    Sp = np.array([[0.,1],[0,0]])
    Sm = np.array([[0.,0],[1,0]])
    I  = np.eye(2)
    O  = np.zeros([2,2])

    Dw = 3*L_row+2 if L_col > 1 else 5
    H = []

    # make parameters iterable if needed
    h      = _make_iterable(h     ,  L_row    *  L_col    )
    J_row  = _make_iterable(J_row , (L_row-1) *  L_col    )
    Jz_row = _make_iterable(Jz_row, (L_row-1) *  L_col    )
    J_col  = _make_iterable(J_col ,  L_row    * (L_col-1) )
    Jz_col = _make_iterable(Jz_col,  L_row    * (L_col-1) )

    # perform consistency checks on parameters
    assert len(h)      ==  L_row    *  L_col   , (
        _assert_parameter_length.format('h'     ,  L_row    *  L_col   ))

    assert len(J_row)  == (L_row-1) *  L_col   , (
       _assert_parameter_length.format('J_row' , (L_row-1) *  L_col   ))

    assert len(Jz_row) == (L_row-1) *  L_col   , (
       _assert_parameter_length.format('Jz_row', (L_row-1) *  L_col   ))

    assert len(J_col)  ==  L_row    * (L_col-1), (
       _assert_parameter_length.format('J_col' ,  L_row    * (L_col-1)))

    assert len(Jz_col) ==  L_row    * (L_col-1), (
       _assert_parameter_length.format('Jz_col',  L_row    * (L_col-1)))

    # first site
    site = np.zeros([Dw,1,2,2])

    site[0,-1] = -h[0] * Sz

    if L_row > 1:
        site[1,-1] =  -J_row[0] * Sm/2 # Sp Sm
        site[2,-1] =  -J_row[0] * Sp/2 # Sm Sp
        site[3,-1] = -Jz_row[0] * Sz   # Sz Sz

    if L_col > 1:
        site[1+(L_row-1)*3,-1] = -J_col[0]  * Sm/2 # Sp I ... I Sm
        site[2+(L_row-1)*3,-1] = -J_col[0]  * Sp/2 # Sm I ... I Sp
        site[3+(L_row-1)*3,-1] = -Jz_col[0] * Sz   # Sz I ... I Sz

    # implement boundary conditions in L_row direction
    if cylinder and L_row > 2:
        site[1+(L_row-2)*3,-1] = 1 * Sm/2 # Sp I ... I Sm
        site[2+(L_row-2)*3,-1] = 1 * Sp/2 # Sm I ... I Sp
        site[3+(L_row-2)*3,-1] = 1 * Sz   # Sz I ... I Sz

    site[-1,-1] = I
    site = site.swapaxes(0,1)
    H.append(site)

    # bulk sites
    J_row_counter = 1
    J_col_counter = 1
    for i in range(1,L_row*L_col-1):

        site = np.zeros([Dw,Dw,2,2])
        site[0,-1]  = -h[i] * Sz
        site[0,0]   = I
        site[0,1]   = Sp # Sp I ... I Sm and Sp Sm
        site[0,2]   = Sm # Sm I ... I Sp and Sm Sp
        site[0,3]   = Sz # Sz I ... I Sz and Sz Sz
        site[-1,-1] = I

        if L_col > 1:

            # build identity ladder for
            # nearest neighbor interactions in L_col direction
            for l in range(L_row-1):
                site[1+l*3,1+(l+1)*3] = I # Sp I ... I Sm
                site[2+l*3,2+(l+1)*3] = I # Sm I ... I Sp
                site[3+l*3,3+(l+1)*3] = I # Sz I ... I Sz

            # skip J_col on last column
            if i // L_row < L_col - 1:

                # Sp I ... I Sm
                site[1+(L_row-1)*3,-1] =  -J_col[J_col_counter] * Sm/2

                # Sm I ... I Sp
                site[2+(L_row-1)*3,-1] =  -J_col[J_col_counter] * Sp/2

                # Sz I ... I Sz
                site[3+(L_row-1)*3,-1] = -Jz_col[J_col_counter] * Sz

                J_col_counter += 1

        # implement boundary conditions for the rows
        if cylinder and L_row > 2 and i % L_row == 0:
            site[1+(L_row-2)*3,-1] = 1 * Sm/2 # Sp I ... I Sm
            site[2+(L_row-2)*3,-1] = 1 * Sp/2 # Sm I ... I Sp
            site[3+(L_row-2)*3,-1] = 1 * Sz   # Sz I ... I Sz

        # nearest neighbor interactions in L_row direction
        if not (i % L_row == L_row - 1):
            site[1,-1] =  -J_row[J_row_counter] * Sm/2 # Sp Sm
            site[2,-1] =  -J_row[J_row_counter] * Sp/2 # Sm Sp
            site[3,-1] = -Jz_row[J_row_counter] * Sz   # Sz Sz
            J_row_counter += 1

        site = site.swapaxes(0,1)
        H.append(site)

    # last site
    site = np.zeros([1,Dw,2,2])
    site[0,0] = I
    site[0,1] = Sp
    site[0,2] = Sm
    site[0,3] = Sz
    site[0,-1] = -h[-1] * Sz
    site = site.swapaxes(0,1)
    H.append(site)

    return H

def XXZ_Heisenberg_2D_lattice_new(L_row,L_col,J_row,Jz_row,J_col,Jz_col,h):
    '''
    Returns the Hamiltonian for the two-dimensional XXZ Heisenberg lattice for
    given side lengths L_row and L_col and given parameters J_row, Jz_row,
    J_col, Jz_col and h. It is advised to set L_row <= L_col in order to
    receive MPOs with smaller local bond dimensions. The returned MPO contains
    no symmetries. If the arguments J_row, Jz_row, J_col, Jz_col and h are
    given as numbers, the J_row_k, Jz_row_k, J_col_k, Jz_col_k and h_k for each
    site are set to this value. Open boundary conditions are assumed.

    The parameters can also be given as two-dimensional arrays, where the
    position inside the array represents the position of the site or bond in
    the lattice. Open boundary conditions are assumed. The shape of each array
    is the following:

    J_row  | ( (L_row-1) ,  L_col    )
    Jz_row | ( (L_row-1) ,  L_col    )
    J_col  | (  L_row    , (L_col-1) )
    Jz_col | (  L_row    , (L_col-1) )
    h      | (  L_row    ,  L_col    )

    H = -\sum_{<k,l>} J (+_k -_l + -_k +_l) + Jz z_k z_l
        -\sum_{k=1}^{L} h_k z_k

    Form of the snake:

    o   o   o   o   o
    |  /|  /|  /|  /|
    o / o / o / o / o
    |/  |/  |/  |/  |
    o   o   o   o   o

    This function is essentially the same as XXZ_Heisenberg_2D_lattice but with
    a changed interpretation of the parameters and a lack of support for
    cylindrical boundary conditions.
    '''

    # set 2x2 matrices
    Sx = 0.5*np.array([[0,1],[1,0]])
    Sy = 0.5*np.array([[0,-1j],[1j,0]])
    Sz = 0.5*np.array([[1,0],[0,-1]])
    Sp = np.array([[0.,1],[0,0]])
    Sm = np.array([[0.,0],[1,0]])
    I  = np.eye(2)
    O  = np.zeros([2,2])

    # make parameters iterable if needed
    h      = _make_iterable(h     ,  (L_row  ,L_col  ))
    J_row  = _make_iterable(J_row ,  (L_row-1,L_col  ))
    Jz_row = _make_iterable(Jz_row,  (L_row-1,L_col  ))
    J_col  = _make_iterable(J_col ,  (L_row  ,L_col-1))
    Jz_col = _make_iterable(Jz_col,  (L_row  ,L_col-1))

    # perform consistency checks on parameters
    assert h.shape      == (L_row  , L_col ) , (
        _assert_parameter_length.format('h'     , (L_row  ,L_col)  ))
    assert J_row.shape  == (L_row-1, L_col ) , (
        _assert_parameter_length.format('J_row' , (L_row-1,L_col)  ))
    assert Jz_row.shape == (L_row-1,L_col  ) , (
        _assert_parameter_length.format('Jz_row', (L_row-1,L_col)  ))
    assert J_col.shape  == (L_row  ,L_col-1) , (
        _assert_parameter_length.format('J_col' , (L_row  ,L_col-1)))
    assert Jz_col.shape == (L_row  ,L_col-1) , (
        _assert_parameter_length.format('Jz_col', (L_row  ,L_col-1)))

    Dw = 3*L_row+2 if L_col > 1 else 5
    H = []

    # first site
    site = np.zeros([Dw,1,2,2])

    site[0,-1] = -h[0,0] * Sz

    if L_row > 1:
        site[1,-1] =  -J_row[0,0] * Sm/2 # Sp Sm
        site[2,-1] =  -J_row[0,0] * Sp/2 # Sm Sp
        site[3,-1] = -Jz_row[0,0] * Sz   # Sz Sz

    if L_col > 1:
        site[1+(L_row-1)*3,-1] =  -J_col[0,0] * Sm/2 # Sp I ... I Sm
        site[2+(L_row-1)*3,-1] =  -J_col[0,0] * Sp/2 # Sm I ... I Sp
        site[3+(L_row-1)*3,-1] = -Jz_col[0,0] * Sz   # Sz I ... I Sz

    site[-1,-1] = I
    site = site.swapaxes(0,1)
    H.append(site)

    # bulk sites
    J_row_counter = 1
    J_col_counter = 1
    for i in range(1,L_row*L_col-1):
        C,R = divmod(i,L_row)

        site = np.zeros([Dw,Dw,2,2])
        site[0,-1]  = -h[R,C] * Sz
        site[0,0]   = I
        site[0,1]   = Sp # Sp I ... I Sm and Sp Sm
        site[0,2]   = Sm # Sm I ... I Sp and Sm Sp
        site[0,3]   = Sz # Sz I ... I Sz and Sz Sz
        site[-1,-1] = I

        if L_col > 1:

            # build identity ladder for
            # nearest neighbor interactions in L_col direction
            for l in range(L_row-1):
                site[1+l*3,1+(l+1)*3] = I # Sp I ... I Sm
                site[2+l*3,2+(l+1)*3] = I # Sm I ... I Sp
                site[3+l*3,3+(l+1)*3] = I # Sz I ... I Sz

            # skip J_col on last column
            if i // L_row < L_col - 1:
                site[1+(L_row-1)*3,-1] =  -J_col[R,C] * Sm/2 # Sp I ... I Sm
                site[2+(L_row-1)*3,-1] =  -J_col[R,C] * Sp/2 # Sm I ... I Sp
                site[3+(L_row-1)*3,-1] = -Jz_col[R,C] * Sz   # Sz I ... I Sz
                J_col_counter += 1

        if not (i % L_row == L_row - 1):
            site[1,-1] =  -J_row[R,C] * Sm/2 # Sp Sm
            site[2,-1] =  -J_row[R,C] * Sp/2 # Sm Sp
            site[3,-1] = -Jz_row[R,C] * Sz   # Sz Sz
            J_row_counter += 1

        site = site.swapaxes(0,1)
        H.append(site)

    # last site
    site = np.zeros([1,Dw,2,2])
    site[0,0] = I
    site[0,1] = Sp
    site[0,2] = Sm
    site[0,3] = Sz
    site[0,-1] = -h[-1,-1] * Sz
    site = site.swapaxes(0,1)
    H.append(site)

    return H

def XXZ_Heisenberg_2D_lattice_reduced_MPO(L_row,L_col,
                                          J_row,Jz_row,J_col,Jz_col,h,
                                          lower_density=False,cylinder=False):
    '''
    Returns the Hamiltonian for the two-dimensional XXZ Heisenberg lattice for
    given side lengths L_row and L_col and given parameters J_row, Jz_row,
    J_col, Jz_col and h. It is advised to set L_row <= L_col in order to
    receive MPOs with smaller local bond dimensions. The returned MPO respects
    the present U(1) symmetry and its site tensors are of type
    rt.reducedTensor. If the arguments J_row, Jz_row, J_col, Jz_col and h are
    given as numbers, the J_row_k, Jz_row_k, J_col_k, Jz_col_k and h_k for each
    site are set to this value. Open boundary conditions are assumed.

    The parameters can also be given as a list in which case they contain the
    values for each site, respectively, bond. Open boundary conditions are
    assumed. The length of each list of parameters is the following:

    J_row  | (L_row-1) *  L_col
    Jz_row | (L_row-1) *  L_col
    J_col  |  L_row    * (L_col-1)
    Jz_col |  L_row    * (L_col-1)
    h      |  L_row    *  L_col

    The position of the entries in the parameter list corresponds to the order
    in which the snake encounters the corresponding sites and bonds.

    It is possible to choose cylindrical boundary conditions instead of open
    boundary conditions by setting cylinder=True. For cylindrical boundary
    conditions, the boundary conditions in L_row direction (the short
    direction) turn periodic while the boundary conditions in L_col direction
    (the long direction) remain open. The additional values needed for J_row
    and Jz_row are set to 1 so that the interpretations of the numbers stored
    in J_row and Jz_row remain the same.

    The parameter lower_density may be set to True to reduce the amount of
    numbers stored in the reduced tensors. This, however, increases the number
    of charge sectors that are present. Depending on the precise use case, one
    of the two options might be more efficient.

    H = -\sum_{<k,l>} J (+_k -_l + -_k +_l) + Jz z_k z_l
        -\sum_{k=1}^{L} h_k z_k

    Form of the snake:

    o   o   o   o   o
    |  /|  /|  /|  /|
    o / o / o / o / o
    |/  |/  |/  |/  |
    o   o   o   o   o

    '''

    MPO = XXZ_Heisenberg_2D_lattice(L_row,L_col,J_row,Jz_row,J_col,Jz_col,h,
                                    cylinder)

    q_b0 = [0]

    if L_row == 1 or L_col == 1:
        q_b1 = [0 ,-2, 2, 0, 0]
        q_b2 = [0 ,-2, 2, 0, 0]
    else:
        n = MPO[1].shape[0] - 8
        if lower_density:
            q_b1 = [0,-2,2,0,-2,2,0,*[5+3*i for i in range(n)],0]
            q_b2 = [0,-2,2,0,*[5+3*i for i in range(n)],-2,2,0,0]
        else:
            n = int((MPO[1].shape[0]-2)/3)
            q_b1 = [0,*n*[-2,2,0],0]
            q_b2 = [0,*n*[-2,2,0],0]

    q_a  = [-1, 1]

    MPO[0] = rt.reducedTensor(MPO[0], [q_b0,q_b2,q_a,q_a],[-1,1,1,-1],Q=0)
    for i in range(len(MPO)-2):
        MPO[i+1] = rt.reducedTensor(MPO[i+1],
                                    [q_b1,q_b2,q_a,q_a],[-1,1,1,-1],Q=0)

    MPO[-1] = rt.reducedTensor(MPO[-1],[q_b1,q_b0,q_a,q_a],[-1,1,1,-1],Q=0)

    return MPO

def Nearest_Neighbor_2D_lattice_MPO_Uform(L_row,L_col,coefficient_dict={},
                                          keep_complex_if_given=False):
    '''
    Gives back an MPO for a two-dimensional lattice of dimensions
    (L_row x L_col) which includes nearest-neighbor interactions and onsite
    magnetic fields.

    The coefficients are stored in coefficient_dict, a dictionary which may
    contain the following keys:

    Cxx, Cxy, Cxz, Cyx, Cyy, Cyz, Czx, Czy, Czz,
    Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz,
    x, y, z.

    The first letter (C or R) decides if the coefficients are used for
    interactions between columns or rows of the lattice. The two letters
    followed thereafer specify the type of the interaction with the first
    letter being the Pauli matrix associated with the site with the smaller
    index (above or left of the second site). The concluding three keys store
    the onsite chemical potential in all three directions.

    Column interactions must be given as an array of shape (L_row , L_col-1)
    or, alternatively, as a scalar number which is then interpreted as the
    value for every column interaction.

    Row interactions must be given as an array of shape (L_row-1 , L_col) or
    a scalar.

    Chemical potentials must be given as an array of shape (L_row , L_col) or
    a scalar.

    The MPO is build by linearising the 2D lattice into a 1D chain following
    the U pattern:

    o   o---o   o---o                           o---o---o---o---o
    |   |   |   |   |                                           |
    o   o   o   o   o                           o---o---o---o---o
    |   |   |   |   |                           |
    o   o   o   o   o  (for L_row <= L_col) or  o---o---o---o---o  else
    |   |   |   |   |                                           |
    o   o   o   o   o                           o---o---o---o---o
    |   |   |   |   |                           |
    o---o   o---o   o                           o---o---o---o---o

    This function rewrites the Hamiltonian to fit in an MPO consisting of
    floats if that is possible. For example, a yy interaction consists of the
    imaginary y but the yy term is real. If only real coefficients are given,
    this is turned into a -ww interaction where w = imag(y). If this is not
    desired, this feature can be disabled by setting
    'keep_complex_if_given' to True. If the MPO cannot be written with floats,
    no action is taken.
    '''

    all_C_keys = ['Cxx','Cxy','Cxz','Cyx','Cyy','Cyz','Czx','Czy','Czz']
    all_R_keys = ['Rxx','Rxy','Rxz','Ryx','Ryy','Ryz','Rzx','Rzy','Rzz']
    all_P_keys = ['x','y','z']

    C_shape = ( L_row    , (L_col-1))
    R_shape = ((L_row-1) ,  L_col   )
    P_shape = ( L_row    ,  L_col   )

    cd = copy.deepcopy(coefficient_dict)
    keys = tuple(cd)

    # Remove all unnecessary entries and assert the correct form
    for key in keys:
        if np.size(np.nonzero(cd[key])) == 0:
            del cd[key]
            continue

        if key in all_C_keys:
            cd[key] = _make_iterable(cd[key],C_shape)
            assert cd[key].shape == C_shape, (
                _assert_parameter_length.format(key,C_shape))
        elif key in all_R_keys:
            cd[key] = _make_iterable(cd[key],R_shape)
            assert cd[key].shape == R_shape, (
                _assert_parameter_length.format(key,R_shape))
        elif key in all_P_keys:
            cd[key] = _make_iterable(cd[key],P_shape)
            assert cd[key].shape == P_shape, (
                _assert_parameter_length.format(key,P_shape))
        else:
            raise KeyError("'{}' Not a valid key.".format(key))

    # rotate everything for more efficiency
    if L_row > L_col:

        keys = tuple(cd)
        cd2 = {}
        L_row,L_col = L_col,L_row

        for key in keys:
            if key[0] == 'R':
                cd2['C'+key[1:]] = cd[key].transpose()
            elif key[0] == 'C':
                cd2['R'+key[1:]] = cd[key].transpose()
            else:
                cd2[key] = cd[key].transpose()

        cd = cd2

    # Find out the nature of the interactions
    interaction_keys = ['xx','xy','xz','yx','yy','yz','zx','zy','zz']
    keys_both         = []
    keys_columns_only = []
    keys_rows_only    = []
    keys_potential    = []

    for key in interaction_keys:
        if 'C'+key in cd and 'R'+key in cd:
            keys_both.append(key)
        elif 'C'+key in cd:
            keys_columns_only.append(key)
        elif 'R'+key in cd:
            keys_rows_only.append(key)

    for key in all_P_keys:
        if key in cd:
            keys_potential.append(key)

    IR = len(keys_rows_only)
    IC = len(keys_columns_only)
    IB = len(keys_both)

    # define all necessary matrices
    Sx = np.array([[0,1],[1,0]])
    Sy = np.array([[0,-1j],[1j,0]])
    Sz = np.array([[1,0],[0,-1]])
    Sw = np.imag(Sy)
    Sp = np.array([[0.,1],[0,0]])
    Sm = np.array([[0.,0],[1,0]])
    I  = np.eye(2)
    O  = np.zeros([2,2])
    sigma = {'x':Sx,'y':Sy,'z':Sz,'w':Sw}

    # find out if MPO can be written with float instead of complex
    if not keep_complex_if_given:
        write_as_real = True
        type_char = ['R','C','R','C','']
        for i,type_ in enumerate([keys_rows_only,keys_columns_only,
                                  keys_both,keys_potential]):
            for key in type_:
                if key.count('y') == 1:
                    # for floats: all entries must be imaginary
                    if np.max(np.abs(np.real(cd[type_char[i]+key]))) > 0:
                        write_as_real = False
                else:
                    # for floats: all entries must be real
                    if np.max(np.abs(np.imag(cd[type_char[i]+key]))) > 0:
                        write_as_real = False

    # This part changes the dictionary if we can write the MPO with floats.
    # The dictionary is overwise not touched.
    if not keep_complex_if_given and write_as_real:
        for i,type_ in enumerate([keys_rows_only,keys_columns_only,
                                  keys_both,keys_potential]):
            for key in type_:
                if key.count('y') == 0:

                    # just convert complex to float
                    cd[type_char[i]+key] = np.real(cd[type_char[i]+key])

                elif key.count('y') == 1:

                    # convert complex to float by
                    # taking -imag and change 'y' to 'w'
                    cd[type_char[i]+key.replace('y','w')] = (
                        -np.imag(cd[type_char[i]+key]))

                    del cd[type_char[i]+key]

                elif key.count('y') == 2:

                    # convert complex to float, negate and change 'y' to 'w'
                    cd[type_char[i]+key.replace('y','w')] = (
                        -np.real(cd[type_char[i]+key]))

                    del cd[type_char[i]+key]

        for i in range(IR):
            keys_rows_only[i] = keys_rows_only[i].replace('y','w')
        for i in range(IC):
            keys_columns_only[i] = keys_columns_only[i].replace('y','w')
        for i in range(IB):
            keys_both[i] = keys_both[i].replace('y','w')
        for i in range(len(keys_potential)):
            keys_potential[i] = keys_potential[i].replace('y','w')

    Dw = _get_local_bond_dimensions_Uform(L_row,L_col,IR,IC,IB)

    # Building of the MPO starts here

    H = []

    i = 0
    for C in range(L_col):
        for R in range(L_row):

            site = np.zeros([Dw[i],Dw[i+1],2,2],dtype=np.complex)

            # Write potentials in the lower left corner

            for j in range(len(keys_potential)):
                if C%2 == 0:
                    A = cd[keys_potential[j]][R,C]
                else:
                    A = cd[keys_potential[j]][L_row-R-1,C]
                site[-1,0] += A*sigma[keys_potential[j]]

            # Write unity matrices in the corners
            if i > 0:
                site[0,0] = I

            if i < L_row * L_col -1:
                site[-1,-1] = I


            # Write in the diagonal of unities
            LOU = ((IB+IC)*(2*L_row - 1 -R) if R != 0
                   else (IB+IC)*(2*L_row - 1 -L_row))

            if C == L_col - 1 and R != 0:
                LOU -= L_row*(IB+IC)

            for j in range(LOU):
                if i > 0:
                    site[1+IR+IB+IC+j,1+IR+j] = I


            # Write IR entries
            for j in range(IR):
                if i < L_row * L_col -1 and R < L_row -1:
                    if C%2 == 0:
                        A = cd['R'+keys_rows_only[j]][R,C]
                    else:
                        A = cd['R'+keys_rows_only[j]][L_row-R-2,C]
                    site[-1,j+1] = A * sigma[keys_rows_only[j][0]]
                if i > 0:
                    site[j+1,0] = sigma[keys_rows_only[j][1]]


            # Write IB entries
            for j in range(IB):
                if i < L_row * L_col -1:
                    if R < L_row - 1:
                        # R connection
                        if C%2 == 0:
                            A = cd['R'+keys_both[j]][R,C]
                        else:
                            A = cd['R'+keys_both[j]][L_row-R-2,C]
                    else:
                        # C connection (very top or very bottom)
                        if C%2 == 0:
                            A = cd['C'+keys_both[j]][R,C]
                        else:
                            A = cd['C'+keys_both[j]][L_row-R-1,C]

                    site[-1,IR+j+1] = A*sigma[keys_both[j][0]]
                if i > 0:
                    site[IR+j+1,0] = sigma[keys_both[j][1]]

                if R != L_row-1 and C != L_col-1:
                    if C%2 == 0:
                        A = cd['C'+keys_both[j]][R,C]
                    else:
                        A = cd['C'+keys_both[j]][L_row-R-1,C]

                    site[-1,1+IR+2*IC+2*IB + 2*(IB+IC)*(L_row-2-R) + j] = (
                        A*sigma[keys_both[j][0]])


            # Write IC entries
            for j in range(IC):

                # Write end matrix in first column
                if i > 0:
                    site[IR+IB+j+1,0]  = sigma[keys_columns_only[j][1]]

                # Write start matrix in bottom-most row
                if R != L_row-1 and C != L_col-1:
                    print('test2',i)
                    if C%2 == 0:
                        A = cd['C'+keys_columns_only[j]][R,C]
                    else:
                        A = cd['C'+keys_columns_only[j]][L_row-R-1,C]
                    site[-1,1+IR+2*IC+2*IB + 2*(IB+IC)*(L_row-2-R) +j+IB] = (
                        A*sigma[keys_columns_only[j][0]])

                # Write start matrix in bottom most row
                # for top-most or bottom-most C connections
                if R == L_row-1 and C != L_col-1:

                    if C%2 == 0:
                        A = cd['C'+keys_columns_only[j]][R,C]
                    else:
                        A = cd['C'+keys_columns_only[j]][L_row-R-1,C]
                    site[-1,1+IR+2*IC+2*IB + 2*(IB+IC)*(L_row-2-R) +j+IB] = (
                        A*sigma[keys_columns_only[j][0]])

            H.append(site)
            i += 1

    use_reals = True
    # loop through the sites to find out if it can be done with reals
    for site in H:
        if np.max(np.abs(np.imag(site))) > 0:
            use_reals = False
            break

    if use_reals:
        for i,site in enumerate(H):
            H[i] = np.real(site)

    return H

def Kitaev_chain(L,t,Delta,U,mu,epsilon=0):
    '''
    Gives back the MPO of the Kitaev chain of length L with open boundary
    conditions and the following parameters:

    t       : hopping amplitude between neighboring sites
    Delta   : superconducting gap
    U       : interaction between neighboring sites
    mu      : field term in z-direction
    epsilon : field term in x-direction

    If these arguments are given as numbers, the parameters at each position
    are set to that number. Alternatively, they can also be given as a list in
    which case they contain the values for each site.

    If t = Delta, and all given numbers are floats, the data type of the MPO
    will be a float and not a complex as would be required for t != Delta.

    The local bond dimension varies between 3 and 5 depending on the given
    parameters.

    H =  \sum_{k=1}^{L-1} - (t_k + Delta_k)/2 x_k x_{k+1}
                          - (t_k - Delta_k)/2 y_k y_{k+1}
                          +     U_k           z_k z_{k+1}
        -\sum_{k=1}^{L} mu_k/2 z_k + epsilon_k x_k
    '''

    sx = np.array([[0,1],[1,0.]])
    sy = np.array([[0,-1j],[1j,0.]]) if t != Delta else np.zeros([2,2])
    sz = np.array([[1,0],[0,-1.]])
    I  = np.eye(2)
    O  = np.zeros([2,2])

    H = []

    t       = _make_iterable(t,       L-1)
    Delta   = _make_iterable(Delta,   L-1)
    U       = _make_iterable(U,       L-1)
    mu      = _make_iterable(mu,      L  )
    epsilon = _make_iterable(epsilon, L  )

    assert len(t)       == L-1, _assert_parameter_length.format('t'       ,L-1)
    assert len(Delta)   == L-1, _assert_parameter_length.format('Delta'   ,L-1)
    assert len(U)       == L-1, _assert_parameter_length.format('U'       ,L-1)
    assert len(mu)      == L  , _assert_parameter_length.format('mu'      ,L  )
    assert len(epsilon) == L  , _assert_parameter_length.format('epsilon' ,L  )

    # first site
    H.append(np.array([[-mu[0]*sz/2+epsilon[0]*sx,
                        -(t[0] + Delta[0])*sx/2.,
                        -(t[0] - Delta[0])*sy/2.,
                        +U[0]*sz,
                        I]]))

    # bulk sites
    for i in range(1,L-1):

        H.append(np.array([[I,  O, O, O, O],
                           [sx, O, O, O, O],
                           [sy, O, O, O, O],
                           [sz, O, O, O, O],
                           [-mu[i]*sz/2+epsilon[i]*sx,
                            -(t[i] + Delta[i])*sx/2.,
                            -(t[i] - Delta[i])*sy/2.,
                            +U[i]*sz,
                            I]]))

        if np.max(np.abs(U)) == 0.:
            H[-1] = np.delete(H[-1],3,0)
            H[-1] = np.delete(H[-1],3,1)

        if np.max(np.abs(t - Delta)) == 0.:
            H[-1] = np.delete(H[-1],2,0)
            H[-1] = np.delete(H[-1],2,1)

    # last site
    H.append(np.array([[I],[sx],[sy],[sz],[-mu[-1]*sz/2+epsilon[-1]*sx]]))

    # reduce size of MPO if possible
    # (can be done if some terms are being omitted)
    if np.max(np.abs(U)) == 0.:
        H[0] = np.delete(H[0],3,1)
        H[-1] = np.delete(H[-1],3,0)

    if np.max(np.abs(t - Delta)) == 0.:
        H[0] = np.delete(H[0],2,1)
        H[-1] = np.delete(H[-1],2,0)

    return H

def lambda_chain(L,l,h,V,h_edge=0.):
    '''
    Gives back the MPO of the lambda chain of length L with open boundary
    conditions and the following parameters:

    l      : The prefactors of the zxz-terms.
    h      : The field in x-direction.
    V      : A nearest neigbor interaction in x-direction.
    h_edge : An additional edge field only applied to the two outermost spins
             on either site of the chain.

    If these arguments are given as numbers, the parameters at each position
    are set to that number. Alternatively, they can also be given as a list in
    which case they contain the values for each site.

    The local bond dimension for this MPO is 5.

    H =   \sum_{k=1}^{L-2} - l_k z_k x_{k+1} z_{k+2}
        + \sum_{k=1}^{L-1} - V_k x_k x_{k+1}
        + \sum_{k=1}^{L}   - h_k x_k
        + h_edge (z_{1} + z_{2} + z_{L} + z_{L-1})
    '''

    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    I  = np.eye(2)
    O  = np.zeros([2,2])

    H = []

    l = _make_iterable(l,L-2)
    V = _make_iterable(V,L-1)
    h = _make_iterable(h,L  )

    assert len(l) == L-2, _assert_parameter_length.format('l',L-2)
    assert len(V) == L-1, _assert_parameter_length.format('V',L-1)
    assert len(h) == L  , _assert_parameter_length.format('h',L  )

    # first site
    H.append(np.array([[h[0]*sx+h_edge*sz,O,V[0]*sx,l[0]*sz,I]]))

    # bulk sites
    for i in range(1,L-1):

        hz = h_edge if (i == 1) or (i == len(l)-2) else 0

        H.append(np.array([[I,  O,  O, O, O],
                           [sz, O,  O, O, O],
                           [sx, O,  O, O, O],
                           [O,  sx, O, O, O],
                           [h[i]*sx+hz*sz,O,V[i]*sx,
                            l[i]*sz if i < len(l) else O,I]]))

    # last site
    H.append(np.array([[I],[sz],[sx],[O],[h[-1]*sx+h_edge*sz]]))

    return H


def Hubbard_chain(L,t,Delta,U):
    '''
    The Hubbard chain of length L implemented with local state space
    dimension 4, i.e. as a spin-3/2 chain. Local bond dimension is 6.
    '''

    sp = np.array([[0,1],[0,0]])
    sm = np.array([[0,0],[1,0]])
    sz = 0.5*np.array([[1,0],[0,-1]])

    I4  = np.eye(4)
    I2  = np.eye(2)
    O   = np.zeros([4,4])

    sp1 = np.kron(sp,I2)
    sp2 = np.kron(I2,sp)

    sm1 = np.kron(sm,I2)
    sm2 = np.kron(I2,sm)

    sz1 = np.kron(sz,I2)
    sz2 = np.kron(I2,sz)

    H = []
    sign = +1

    t     = _make_iterable(t,    L)
    Delta = _make_iterable(Delta,L)
    U     = _make_iterable(U,    L)

    assert len(t)     == L-1, _assert_parameter_length.format('t',    L-1)
    assert len(Delta) == L,   _assert_parameter_length.format('Delta',L  )
    assert len(U)     == L,   _assert_parameter_length.format('U',    L  )

    # first site
    H.append(np.array([[U[0]*np.dot(sz1,sz2)+Delta[0]*sign*(sz1+sz2),
                        t[0]*sm1, t[0]*sp1 , t[0]*sm2, t[0]*sp2, I4]]))

    # bulk sites
    for i in range(1,L-1):

        sign = -sign

        H.append(np.array([[I4,  O,  O, O, O, O],
                           [sp1, O,  O, O, O, O],
                           [sm1, O,  O, O, O, O],
                           [sp2, O,  O, O, O, O],
                           [sm2, O,  O, O, O, O],
                           [U[i]*np.dot(sz1,sz2)+Delta[i]*sign*(sz1+sz2),
                            t[i]*sm1 , t[i]*sp1 , t[i]*sm2, t[i]*sp2, I4]]))

    # last site
    sign = -sign
    H.append(np.array([[I4],[sp1],[sm1],[sp2],[sm2],
                       [U[-1]*np.dot(sz1,sz2)+Delta[-1]*sign*(sz1+sz2)]]))

    return H

def field_MPO(L,A,h=1.):
    '''
    An MPO of length 'L' with only an onsite field in direction of matrix A.
    A field strength can be supplied for each site by setting 'h' to a list of
    length 'L' or for all sites together by setting 'h' to a number.
    '''

    I  = np.eye(2)
    O  = np.zeros([2,2])

    h     = _make_iterable(h,L)
    assert len(h) == L, _assert_parameter_length.format('h',L)

    H = []
    
    # first site
    H.append(np.array([[h[0]*A,I]]))

    # bulk sites
    for i in range(1,L-1):
        H.append(np.array([[I,  O],
                           [h[i]*A, I]]))

    # last site
    H.append(np.array([[I],[h[-1]*A]]))
    return H

def X_field_MPO(L,h=1.):
    '''
    An MPO of length 'L' with only an onsite field in x-direction. A field
    strength can be supplied for each site by setting 'h' to a list of
    length 'L' or for all sites together by setting 'h' to a number.
    '''

    sx = np.array([[0,1],[1,0.]])
    return field_MPO(L,A=sx,h=h)

def Y_field_MPO(L,h=1.):
    '''
    An MPO of length 'L' with only an onsite field in y-direction. A field
    strength can be supplied for each site by setting 'h' to a list of
    length 'L' or for all sites together by setting 'h' to a number.
    '''

    sy = np.array([[0,-1j],[1j,0.]])
    return field_MPO(L,A=sy,h=h)

def Z_field_MPO(L,h=1.):
    '''
    An MPO of length 'L' with only an onsite field in z-direction. A field
    strength can be supplied for each site by setting 'h' to a list of
    length 'L' or for all sites together by setting 'h' to a number.
    '''

    sz = np.array([[1,0],[0,-1.]])
    return field_MPO(L,A=sz,h=h)
