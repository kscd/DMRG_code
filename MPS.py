import numpy as np
import scipy as sp
import scipy.linalg
import copy

# import reducedTensor class if it exists
# if not, only dense tensors can be used
try:
    import reducedTensor as rt
    _reducedTensorExists = True
except ImportError:
    _reducedTensorExists = False

def tensordot(tensor1,tensor2,axes):
    '''
    This function is a wrapper to both np.tensordot and
    rt.contract_reduced_tensors_along_axes. Its purpose is to call the function
    appropriate to the type of the supplied tensors. If the tensors are of type
    np.ndarray the function np.tensordot is called, where as
    rt.contract_reduced_tensors_along_axes is called if both tensors are of
    type rt.reducedTensor. The purpose of the function is thus to prevent the
    user of the supplied tensors to require details about the tensors. Both
    tensors must have the same data type.
    '''

    if type(tensor1) is not type(tensor2):
        raise TypeError('tensor1 and tensor2 are not of the same type, tensor1'
                        ' is {} and tensor2 is {}.'.format(type(tensor1),
                                                           type(tensor2)))

    if type(tensor1) is np.ndarray:
        return np.tensordot(tensor1,tensor2,axes)
    elif _reducedTensorExists and type(tensor1) is rt.reducedTensor:
        return rt.contract_reduced_tensors_along_axes(tensor1,tensor2,
                                                      axis1_list=axes[0],
                                                      axis2_list=axes[1])
    else:
        raise TypeError('tensor1 and tensor2 are of type '
                        '{} but must be either a '.format(type(tensor1))+
                        'reduced tensor or a numpy array.')

def perform_QR(matrix):
    '''
    This function is a wrapper to both np.linalg.qr and
    rt.QR_for_reduced_matrix. The respective function is called based on the
    data type of the given matrix. This function thus serves as a unified way
    to perform a QR decomposition without having to worry about data types.
    '''

    if type(matrix) is np.ndarray:
        return np.linalg.qr(matrix)
    elif _reducedTensorExists and type(matrix) is rt.reducedTensor:
        return rt.QR_for_reduced_matrix(matrix)
    else:
        raise TypeError('matrix is of type {} but '.format(type(matrix))+
                        'must be either a reduced tensor or a numpy array.')

def trace(matrix):
    '''
    This function serves as a wrapper to both np.trace and
    rt.trace_for_reduced_matrix. Which function is called to calculate the
    trace of the supplied matrix depends on the data type of said matrix.
    '''

    if type(matrix) is np.ndarray:
        return np.trace(matrix)
    elif _reducedTensorExists and type(matrix) is rt.reducedTensor:
        return rt.trace_for_reduced_matrix(matrix)
    else:
        raise TypeError('matrix is of type {} but '.format(type(matrix))+
                        'must be either a reduced tensor or a numpy array.')


def _check_if_reduced_tensors_compatible(tensor1,tensor2):
    '''
    Check if two supplied tensors of type reducedTensors are compatible with
    each other in the sense that they share the same charge vectors and charge
    signs. This function serves as a consistency check and raises a TypeError
    if the supplied tensors are not compatible.
    '''

    if len(tensor1.q_vectors) != len(tensor2.q_vectors):
        raise TypeError("Tensors don't have the same number of dimensions.")

    for i in range(len(tensor1.q_vectors)):

        if list(tensor1.q_vectors[i]) != list(tensor2.q_vectors[i]):
            raise TypeError('charge vector {} is not equal.'.format(i))

    if list(tensor1.q_signs) != list(tensor2.q_signs):
        raise TypeError('charge signs are not equal.')

def _check_if_reduced_tensor_has_q_vectors(tensor,q_vectors):
    '''
    Check if a given reducedTensor has the same charge vectors as supplied. If
    this is not the case a TypeError will be raised.
    '''

    for q_i in range(len(q_vectors)):
        if list(q_vectors[q_i]) != list(tensor.q_vectors[q_i]):
            raise TypeError('Tensor has not the correct charge vectors.')

def give_valid_charge_sectors(L,S):
    '''
    For a length L and a magnetisation S, a set of charge sector addresses is
    returned which represent the possible charge sectors for each site tensor
    position under the assumption that a local state space dimension of d=2 is
    used. The returned object is a list and each entry is a numpy array. The
    position within the list represents the position in the MPS, while the
    numbers in each array represent the individual charge sector addresses.
    '''

    assert S<=L and S>=-L, 'S must lie in the interval [-L,L].'

    charge_sectors = [np.array([0])]
    for l in range(1,L+1):

        next_sector_set = set(np.concatenate([charge_sectors[l-1]-1,
                                              charge_sectors[l-1]+1]))

        allowed_list = []
        for entry in next_sector_set:
            if entry + L - l >= S and entry - L + l <= S:
                allowed_list.append(entry)

        charge_sectors.append(np.array(allowed_list))

    return charge_sectors

def perform_np_SVD(matrix, D=None, max_truncation_error=None,
                   normalize='S',normalize_to_unity=False, return_='U|S|V'):
    '''
    This function is a wrapper for np.linalg.svd and calculates the singular
    value decomposition for a given matrix given as np.ndarray. np.linalg.svd
    is known to fail at calculating the SVD of a given valid matrix. This
    function tries to compensate for this by modifying the given matrix
    slightly. Afterwards, post-production takes place.

    In the first step the resulting matrices are approximated by smaller
    matrices. For this, either the exact number of singular values to keep, D,
    can be supplied or a maximal truncation error.  If both are set the number
    of kept singular values is at least D, even if the resulting truncation
    error will be larger than desired.

    Afterwards, the results are normalized. The matrix to be normalized is
    given with 'normalize'.  Per default the singular values are rescaled to
    ensure normalisation but 'U' and 'V' can also be supplied. All other
    choices will result in no rescaling. The normalisation preserves the norm
    of the given matrix.

    The function returns the number of kept singular values and the truncation
    error. What else is returned is supplied with 'return_'. Per default U, S
    and V are returned. The returned matrices are seperated with pipes, |, and
    any valid matrix product can be supplied. 's' corresponds to np.sqrt(S) and
    single occuring 'S' and 's' are given as one-dimensional arrays instead of
    two-dimensional ones.

    Example: matrix is a matrix of size 50x50, D = 25, normalize = 'S',
    return_='Us|sV|S' The matrix is SVDed and cut to 25 singular values.
    This function returns: D, truncation_error, np.dot(U,np.sqrt(S)),
    np.dot(np.sqrt(S),V), np.diag(S).
    '''

    # perform SVD with different techniques to
    # compensate for bugs in LAPACK '_gesdd'

    failure = False

    # Try regular svd.
    try:
        U,S,V = np.linalg.svd(matrix, full_matrices=False, compute_uv=True)
    except np.linalg.LinAlgError:
        failure = True

    # If calculation of SVD failed, try again with
    # matrix rounded to 15 decimal digits
    if failure:
        print('SVD did not converge. '
              'Trying again with matrix rounded to 15 decimal digits.')
        try:
            U,S,V = np.linalg.svd(np.round(matrix,15), full_matrices=False,
                                  compute_uv=True)
            failure = False
        except np.linalg.LinAlgError:
            pass # so that other errors get raised

    # If calculation of SVD failed again, try again with noise of order 1e-15.
    # Try this 10 times, afterwards raise np.linalg.LinAlgError.
    if failure:
        for i in range(10):
            print('SVD did not converge. '
                  'Try again with random noise. Try {}/10.'.format(i+1))
            try:
                U,S,V = np.linalg.svd(matrix +
                                      (np.random.random(np.shape(matrix))-
                                       0.5)*1e-15, full_matrices=False,
                                      compute_uv=True)
                failure = False
                break
            except np.linalg.LinAlgError as e:
                # At one point we just have to abbort
                if i == 9:
                    raise e

    # compress resulting matrices
    old_norm = np.sum(S**2)
    D = len(S) if D is None else min(D,len(S))

    # if max_truncation_error is set, find new new D. (Yes, two 'new's)
    if max_truncation_error is not None:

        for i in range(D - 1,0,-1):

            truncation_error = np.sum(S[i:]**2)/old_norm
            if truncation_error > max_truncation_error:
                D = i+1
                break

            if i == 1:
                D = 1

    # correct D if set to a value larger than possible.
    D = min(D,len(S))

    # shrink matrices and normalize
    new_norm         = np.sum(S[:D]**2)
    truncation_error = np.sum(S[D:]**2)/old_norm

    if normalize_to_unity:
        U = U[:,:D] if normalize != 'U' else U[:,:D] / np.sqrt(new_norm)
        S = S[:D]   if normalize != 'S' else S[:D]   / np.sqrt(new_norm)
        V = V[:D]   if normalize != 'V' else V[:D]   / np.sqrt(new_norm)
    else:
        U = U[:,:D] if normalize != 'U' else (U[:,:D] / np.sqrt(new_norm) *
                                              np.sqrt(old_norm))
        S = S[:D]   if normalize != 'S' else (S[:D]   / np.sqrt(new_norm) *
                                              np.sqrt(old_norm))
        V = V[:D]   if normalize != 'V' else (V[:D]   / np.sqrt(new_norm) *
                                              np.sqrt(old_norm))

    # create return list and return results
    return_list = []
    return_dict = {'U':U,'S':S,'V':V}

    for r in return_.split('|'):

        first = True
        for c in r:

            if first:
                if c == 'U':
                    return_list.append(U)
                elif c == 'S':
                    return_list.append(S if len(r) == 1 else np.diag(S))
                elif c == 's':
                    return_list.append(np.sqrt(S) if len(r) == 1
                                       else np.diag(np.sqrt(S)))
                elif c == 'V':
                    return_list.append(V)

            else:
                if c == 'U':
                    return_list[-1]=np.dot(return_list[-1],U)
                elif c == 'S':
                    return_list[-1]=np.dot(return_list[-1],np.diag(S))
                elif c == 's':
                    return_list[-1]=np.dot(return_list[-1],np.diag(np.sqrt(S)))
                elif c == 'V':
                    return_list[-1]=np.dot(return_list[-1],V)

            first = False

    return (D,truncation_error,*return_list)

def reduced_MPS_from_basis_state(d,D,spin_list):
    '''
    Create an MPS object containing site tensors of type reducedTensors from a
    supplied basis state. 'd' is the local state space dimension, while 'D' is
    the local bond dimension. The encoded basis state could be written with D=1
    but setting D to a larger value may be advantageous, for example to perform
    a one-site DMRG calculation. 'spin_list' is a list or other iterable object
    giving for each site the direction the spin is pointing in. The entries in
    'spin_list' must be integers as only basis states in the classical basis
    are possible with this function. The length of the system in question is
    determined by the length of 'spin_list'.
    '''

    if not _reducedTensorExists:
        raise ImportError("'reducedTensor' could not be imported.")

    # create empty MPS (L=0)
    system = MPS.empty_MPS(d,D)

    # define initial charge vectors
    q_sigma  = [-1,1]
    q_a_left = [0]

    # iterate through the chain
    for spin in spin_list:

        # define charge vector for the right side
        q_a_right = [q_a_left[0]-q_sigma[spin]]

        # set properties of site tensor
        sectorname = (q_sigma[spin],q_a_left[0],q_a_right[0])
        Q = 0

        # create the new site tensor at current position
        system._MPS.append(rt.reducedTensor(tensor={sectorname:
                                                    np.array([[[1.]]])},
                                            list_of_q=[q_sigma,q_a_left,
                                                       q_a_right],
                                            list_of_xi=[1,-1,1],
                                            Q=Q,
                                            sectors_given=True))

        # right charge vector of current site must
        # be left charge vector of next site
        q_a_left = q_a_right

    # set properties of MPS
    system._L = len(spin_list)
    system._B_border = len(spin_list)
    system._useReducedTensors = True

    return system


def load_MPS_hdf5(hdf5_handler,load_as_MPS_system=False):
    '''
    Load an MPS from a given hdf5_handler (from h5py). The handler must
    point to the group under which name the MPS was originally saved.
    This works whether reduced tensors are used or not.
    '''

    MPS_ = MPS.empty_MPS(hdf5_handler.attrs['_d'],hdf5_handler.attrs['_D'])
    MPS_._L = hdf5_handler.attrs['_L']

    MPS_._multiplicative_factor = hdf5_handler.attrs['_multiplicative_factor']
    MPS_._normalisation_tolerance = (hdf5_handler.
                                     attrs['_normalisation_tolerance'])
    MPS_.initial_truncation_error = (hdf5_handler.
                                     attrs['initial_truncation_error'])
    MPS_._useReducedTensors = hdf5_handler.attrs['_useReducedTensors']
    MPS_._A_border = hdf5_handler.attrs['_A_border']
    MPS_._B_border = hdf5_handler.attrs['_B_border']
    MPS_._boundary = hdf5_handler.attrs['_boundary']

    if hdf5_handler.attrs['_useReducedTensors']:
        MPS_._MPS = [rt.load_reducedTensor_hdf5(hdf5_handler[
            'site{}'.format(i)]) for i in range(hdf5_handler.attrs['_L'])]
    else:
        MPS_._MPS = [np.array(hdf5_handler[
            'site{}'.format(i)]) for i in range(hdf5_handler.attrs['_L'])]

    return MPS_

def save_MPS_hdf5(hdf5_handler,MPS_,name):
    '''
    Saves a given MPS in a HDF5 file to a given position. This function
    delegates to the function 'save_hdf5' of the MPS object. Saving the MPS to
    HDF5 file is possible regardless of the use of reducedTensor.
    hdf5_handler: The handler to the HDF5 file (from h5py). It can point
                  to a group within the file.
    MPS_:         The MPS object to be saved.
    name:         A new group under this name will be created. In this group
                  the MPS will be saved. This group has to be referenced
                  in the loading function to load the MPS again.
    '''

    MPS_.save_hdf5(hdf5_handler,name)

def _give_convenient_numpy_float_complex_type(type1,type2):
    '''
    Takes two data types which are either float{16,32,64,128} or
    complex{64,128,256} and gives back the best matching data type for both.
    The resulting time is small enough while preventing data loss. Example:
    type1 = np.float128, type2 = np.complex64, returns np.complex256. 'complex'
    because one input was complex and '256' to fit float128 inside.

    '''

    if np.issubdtype(type1, np.integer):
        type1 = np.float64

    if np.issubdtype(type2, np.integer):
        type2 = np.float64

    complex_dict = {np.complex64:32,np.complex128:64,np.complex256:128}
    float_dict   = {np.float16:16,np.float32:32,np.float64:64,np.float128:128}

    complex2_dict = {32:np.complex64,64:np.complex128,128:np.complex256}
    float2_dict   = {16:np.float16,32:np.float32,64:np.float64,128:np.float128}

    use_complex = False
    if type1 in complex_dict:
        use_complex = True
        bit_length1 = complex_dict[type1]
    else:
        bit_length1 = float_dict[type1]

    if type2 in complex_dict:
        use_complex = True
        bit_length2 = complex_dict[type2]
    else:
        bit_length2 = float_dict[type2]

    if use_complex:
        return complex2_dict[max(bit_length1,bit_length2)]
    else:
        return float2_dict[max(bit_length1,bit_length2)]


class MPS:
    '''
    This class represents a matrix product state (MPS) of either open or
    periodic boundary conditions and may use as site tensors either instances
    of np.ndarray or rt.reducedTensor. As a special form some quantum state may
    efficiently be written in, the usage of matrix product states require a
    reimplementation of a number of operation including but not limited to
    overlaps, matrix elements or superpositions. This class provides a number
    of these functionalities in order to make the usage of MPSs more versatile.
    Due to the fact that MPSs are the backbone of a number of tensor network
    algorithms such as the DMRG or the tMPS, this class comes equipped with a
    number of functionalities to ease the implementation and usage of such
    methods.
    '''

    ##################################################################
    # Possible features which could be included to enhance the class #
    ##################################################################
    # Support for N-site tensors (currently only one and two)        #
    # Support for periodic boundary conditions (some work done)      #
    # Shrinking the MPS (i.e. deleting sites)                        #
    # Compressing the MPS (variational)                              #
    ##################################################################

    ## The following functions don't perform proper checks on ingoing
    ## reduced tensors and might not even work:
    ## set_several_site_tensors

    def __init__(self,d,L,D,initialise='const',boundary='OBC',*,const=0.,
                 spin_list=None,state_vector=None,
                 state_vector_truncation_error=None):
        '''
        Initialises the MPS class. There are several ways to initialise this
        class based on individual needs. The precise form the initialisation
        takes is determined by the variable 'initialise' which may take the
        following values:
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

        self._d = d
        self._L = L
        self._D = D

        self._MPS = []

        self._multiplicative_factor = 1.0
        self._normalisation_tolerance = 1e-14
        self.initial_truncation_error = 0
        self._boundary = 'OBC'

        self._useReducedTensors = False

        if initialise in ['const','constant']:
            self._set_MPS_to_constant_values(const)
        elif initialise in ['basis','basis state','basis_state']:
            self._set_MPS_from_basis_state(spin_list)
        elif initialise in ['vector','state vector','state_vector']:
            x = self._set_MPS_from_state_vector(state_vector,
                                                state_vector_truncation_error)

            self.initial_truncation_error = x[0]
            self._multiplicative_factor = x[1]
        elif initialise in ['empty']:
            self._A_border = 0
            self._B_border = 0

    # magic methods

    def __len__(self):
        '''
        Returns the length of the MPS.
        '''

        return self._L

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

        # initial checks
        if self._L != MPS._L:
            raise ValueError('Both MPS need to be of the same length.')

        if self._d != MPS._d:
            raise ValueError('Both MPS need to have the same '
                             'local state space dimension.')

        # create empty MPS and evaluate boundary conditions
        MPSsum = MPS.empty_MPS(self._d,self._D)

        openBC = (True if self.get_boundary_condition() == 'OBC' and
                  MPS.get_boundary_condition() == 'OBC' else False)

        # loop through site tensors and concatenate them accordingly
        for site in range(self._L):

            tensor1 = self.get_site(site)
            tensor2 = (MPS.get_site(site) if site > 0
                       else factor*MPS.get_site(site))

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
            new_type = _give_convenient_numpy_float_complex_type(tensor1.
                                                                 dtype.type,
                                                                 tensor2.
                                                                 dtype.type)

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

            MPSsum.insert_site_tensor(site,new_tensor,different_shape=True)

        return MPSsum

    def __add__(self,MPS):
        '''
        Adds this MPS and the given MPS together. Returns a new MPS instance
        which stores the sum of the two MPSs.
        '''

        return self._add_sub_magic_method(MPS,+1.)

    def __sub__(self,MPS):
        '''
        Subtracts the given MPS from this MPS. Returns a new MPS instance which
        stores the sum of the two MPSs.
        '''

        return self._add_sub_magic_method(MPS,-1.)

    def __iadd__(self,MPS):
        '''
        Adds the given MPS inplace to this MPS.
        '''

        self.add(MPS)
        return self

    def __imul__(self,factor):
        '''
        Multiplies the given factor inplace to this MPS. This is done by
        modifying the very first site tensor accordingly.
        '''

        self.set_site(0,factor*self.get_site(0))
        return self

    def __itruediv__(self,divisor):
        '''
        Multiplies the given factor inplace to this MPS. This is done by
        modifying the very first site tensor accordingly.
        '''

        self.set_site(0,self.get_site(0)/divisor)
        return self

    ########################################################################
    # Secondary constructors: provide a more fitted interface than __init__#
    ########################################################################

    @classmethod
    def constant_MPS(cls,d,L,D,const):
        '''
        Creates an MPS with local bond dimension self._D and fills it
        completely with the 'const'.
        '''

        return cls(d,L,D,initialise='const',const=const)

    @classmethod
    def zero_MPS(cls,d,L,D):
        '''
        Creates an MPS with local bond dimension self._D and fills it
        completely with zeros.
        '''

        return cls(d,L,D,initialise='const',const=0)

    @classmethod
    def MPS_from_basis_state(cls,d,D,spin_list):
        '''
        Creates a basis state, meaning that one entry in the state vector
        is one, the rest zero. The selected state is given by the list of spin
        directions. If for example d=2, spin_list could be [0,0,1,1,0,1]
        meaning that the first two spins point in one direction, the following
        two in the other and so forth.
        '''

        L = len(spin_list)
        return cls(d,L,D,initialise='basis state',spin_list=spin_list)

    @classmethod
    def MPS_from_random_basis_state(cls,d,L,D):
        '''
        Creates an MPS from a random basis state, where the probability of the
        direction each spin is pointing at is equal.
        '''

        spin_list = np.random.randint(low=0,high=d,size=L)
        return cls(d,L,D,initialise='basis state',spin_list=spin_list)

    @classmethod
    def empty_MPS(cls,d,D):
        '''
        Creates an MPS of size zero. No site tensors exist here. This can be
        used to initialise the MPS object for an algorithm which then grows the
        MPS.
        '''

        return cls(d,0,D,initialise='empty')

    @classmethod
    def MPS_from_state_vector(cls,d,L,D,state_vector,
                              state_vector_truncation_error=None):
        '''
        Creates a left-normalized MPS from a given state tensor. Note that the
        tensor grows exponentially with the number of sites meaning that this
        method is only suitable for small toy models.
        '''

        return cls(d,L,D,initialise='state vector',state_vector=state_vector,
                   state_vector_truncation_error=state_vector_truncation_error)

    ##########################################################################
    # Functions __init__ and the sec. constructors rely on to create the MPS #
    ##########################################################################

    def _set_MPS_to_constant_values(self,const):
        '''
        Creates an MPS with local bond dimension self._D and fills it
        completely with 'const'.
        '''

        const = float(const)

        L = self._L
        d = self._d

        D_left = 1
        D_right = 2

        # create first half
        for i in range(int(L/2)):
            D_left  = min(self._D, d**i)
            D_right = min(self._D, d**(i+1))

            self._MPS.append(np.full([d,D_left,D_right],const))

        # create second half
        if L % 2 == 1:
            D  = min(self._D, d**int(L/2))

            self._MPS.append(np.full([d,D,D],const))

        for i in range(int(L/2),0,-1):
            D_left  = min(self._D, d**i)
            D_right = min(self._D, d**(i-1))

            self._MPS.append(np.full([d,D_left,D_right],const))

        self._A_border = 0
        self._B_border = self._L

    def _set_MPS_from_basis_state(self, spin_list):
        '''
        Creates a basis state, meaning that one entry in the state vector
        is one, the rest zero. The selected state is given by the list of spin
        directions.
        '''

        self._set_MPS_to_constant_values(0.0)

        for i in range(len(spin_list)):
            self._MPS[i][spin_list[i],0,0] = 1.0

    def _set_MPS_from_state_vector(self, state_tensor,
                                   state_vector_truncation_error):
        '''
        Creates a left-normalized MPS from a given state tensor. Note that the
        tensor grows exponentially with the number of sites meaning that this
        method is only suitable for small toy models.
        '''

        dimC = np.shape(state_tensor)
        matrix = state_tensor.reshape(dimC[0],int(np.prod(dimC[1:])))

        sum_trunc_err = 0

        D_here = min(int(self._d**(self._L/2)),self._D)

        edge_size = int(np.floor(np.log(D_here)/np.log(self._d)))

        for i in range(self._L):

            if i < edge_size:
                D_here = self._d ** (i+1)
            elif i > self._L-edge_size-1:
                D_here = self._d ** (self._L - i - 1)
            else:
                D_here = self._D

            # perform SVD
            U,S,V = np.linalg.svd(matrix, full_matrices=False, compute_uv=True)

            # if state_vector_truncation_error is set, find new D_here.
            if state_vector_truncation_error is not None:

                for i in range(D_here-1,1,-1):

                    truncation_error = np.sum(S[i:]**2)
                    if truncation_error > max_truncation_error:
                        D_here = i+1
                        break

                    if i == 1:
                        D_here = 1

            # shrink the matrices
            old_norm         = np.linalg.norm(S)
            new_norm         = np.linalg.norm(S[:D_here])
            truncation_error = 1 - new_norm**2/old_norm**2
            sum_trunc_err   += truncation_error

            S = S[:D_here]/new_norm*old_norm #normalisation
            U = U[:,:D_here]
            V = V[:D_here]

            # shape U into new A
            dim2 = int(np.shape(U)[0]/self._d)
            dim3 = np.shape(U)[1]
            U = U.reshape((dim2, self._d, dim3))
            U = U.swapaxes(0,1)
            self._MPS.append(U)

            # shape SV into new matrix
            matrix = np.dot(np.diag(S),V)

            if np.size(matrix) != 1:

                dim1 = dim3*self._d
                dim2 = int(np.size(matrix)/(dim3*self._d))

                matrix = matrix.reshape(np.size(matrix))
                matrix = matrix.reshape((dim1,dim2))

        norm = matrix[0,0]

        self._A_border = self._L
        self._B_border = self._L

        return [sum_trunc_err,norm]

    ################################################
    # Functions to receive attributes from the MPS #
    ################################################

    def get_length(self):
        '''
        Returns the length of the MPS.
        '''

        return self._L

    def get_local_bond_dimension(self):
        '''
        Returns the local bond dimension of the MPS.
        '''

        return self._D

    def get_local_state_space_dimension(self):
        '''
        Returns the local state space dimension of the MPS.
        '''

        return self._d

    def get_boundary_condition(self):
        '''
        Returns the boundary condition imposed on the MPS.
        '''

        return self._boundary

    def get_left_normalized_border(self):
        '''
        Returns the position where all site tensors left of it are
        left-normalized.
        '''

        self._check_for_open_boundary_conditions()
        return self._A_border

    def get_right_normalized_border(self):
        '''
        Returns the position where all site tensors right of it are
        right-normalized including returned position.
        '''

        self._check_for_open_boundary_conditions()
        return self._B_border

    def get_local_bond_dimensions(self):
        '''
        Returns a one-dimensional numpy array which shows the local bond
        dimensions between the site tensors, beginning on the left edge and
        ending at the right edge.
        '''

        D = np.empty([self.get_length()+1],dtype=np.uint32)

        for i in range(self.get_length()):
            D[i] = np.shape(self.get_site(i))[1]

        D[self.get_length()] = np.shape(self.get_site(self.get_length()-1))[2]
        return D

    ##########################################
    # Functions to set attributes of the MPS #
    ##########################################

    def set_local_bond_dimension(self,D):
        '''
        Change the local bond dimension to a new value.
        '''

        if type(D) is not int:
            raise TypeError('The local bond dimension must be an integer.')

        if D < 1:
            raise ValueError('The local bond dimension must be at least 1.')

        self._D = D

    def set_boundary_condition(self,bc):
        '''
        Set the boundary condition of the chain. It must be either open ('OBC')
        or closed ('PBC').
        '''

        if bc in 'OBC':
            self._boundary = 'OBC'
        elif bc in 'PBC':
            self._boundary = 'PBC'
        else:
            raise ValueError("The given boundary condition must be either "
                             "'OBC' or 'PBC' but is {}.".format(bc))

    ###################################################
    # Functions involving the site tensors of the MPS #
    ###################################################

    def get_site(self,pos):
        '''
        This function returns one particular site tensor counting from zero.
        Negative indices are allowed. Raises a ValueError if non-existing
        element is tried to be reached. The index order of the given site
        tensor is: s_p | a_p-1 | a_p.
        '''

        self._check_if_position_is_valid(pos)
        return self._MPS[pos]

    def get_site_from_right(self,pos_from_right):
        '''
        The same as get_site() but counts from the right. Negative indexing is
        also allowed here. The index order of the given site tensor is:
        s_p | a_p-1 | a_p.
        '''

        pos = -pos_from_right-1
        return self.get_site(pos)

    def set_site(self,pos,site_tensor,different_shape=False):
        '''
        Change a site tensor at a particular position to the supplied site
        tensor. The new site tensor will be treated as being not normalized.
        This affects the position of the A- and B-boundaries. If the tensor is
        meant to have a different shape or different charge vectors if reduced
        tensors are used, it must be explicitly mentioned by setting
        different_shape=True to deactivate compatibility tests. By changing the
        shape or the charge vector structure, the MPS becomes illegal. The user
        has to ensure that further operations on the MPS makes the MPS legal
        again. The index order of the tensor 'site_tensor'
        must be: s_p | a_p-1 | a_p.
        '''

        # perform consistency checks
        self._check_if_position_is_valid(pos)

        self._check_if_array(site_tensor)

        self._check_if_tensor_has_right_number_of_dimensions(site_tensor,3)

        # check if 'site_tensor' is compatible
        shape_old = np.shape(self.get_site(pos))
        shape_new = np.shape(site_tensor)

        if shape_old != shape_new and not different_shape:
            raise ValueError('New tensor has not the same site as the old '
                             'one. This would lead to an illegal MPS.')

        if self._useReducedTensors and not different_shape:
            _check_if_reduced_tensors_compatible(site_tensor,
                                                 self.get_site(pos))

        # lastly, set the new tensor
        self._MPS[pos] = site_tensor

        # Don't do anything further for non-open boundary conditions
        if self.get_boundary_condition() != 'OBC':
            return

        # for open boundary conditions: change A and B boundary

        # force indices to be positive
        if pos < 0:
            pos = pos + self.get_length()

        # A boundary
        if self.get_left_normalized_border() > pos:
            self._A_border = pos

        # B boundary
        if self.get_right_normalized_border() < pos+1:
            self._B_border = pos+1

    def set_site_from_right(self,pos_from_right,site_tensor):
        '''
        Change a site tensor at a particular position from the right to the
        supplied site tensor. This function is a wrapper to 'set_site'. The
        index order of the tensor 'site_tensor' must be: s_p | a_p-1 | a_p.
        '''

        pos = -pos_from_right-1
        self.set_site(pos,site_tensor)

    def check_if_site_left_normalized(self,pos):
        '''
        Checks if a given site tensor is left-normalized or not. For this
        purpose, the largest deviation of dagger(A)@A from the identity matrix
        is checked against the normalisation tolerance. If it is smaller than
        this tolerance, the site tensor A is declared left-normalised.
        '''

        A_tensor = self.get_site(pos)

        # index order: s_p | a'_p | a_p-1 @ s_p | a_p-1 | a_p => a'_p,a_p
        matrix = tensordot(self._dagger(A_tensor), A_tensor, ((0, 2), (0, 1)))

        # return whether maximum deviation from identity matrix
        # is below normalisation tolerance
        if type(matrix) is np.ndarray:

            # for dense numpy tensors
            dim_matrix = np.shape(matrix)[0]
            matrix = matrix - np.identity(dim_matrix)
            return np.max(np.abs(matrix)) < self._normalisation_tolerance

        elif self._useReducedTensors and type(matrix) is rt.reducedTensor:

            # for sparse rt.reducedTensor tensors
            # charge vectors are equal for both dimensions

            max_err = 0
            for sector in matrix.sectors.keys():
                submatrix = matrix.sectors[sector]
                dim_matrix = np.shape(submatrix)[0]
                submatrix = submatrix - np.identity(dim_matrix)
                max_err = max( max_err , np.max(np.abs(submatrix)) )
            return max_err < self._normalisation_tolerance

        else:
            raise TypeError('The site tensor at position {} '.format(pos)+
                            'is neither an np.ndarray or a rt.reducedTensor.')

    def check_if_site_right_normalized(self,pos):
        '''
        Checks if a given site tensor is right-normalized or not. For this
        purpose, the largest deviation of B@dagger(B) from the identity matrix
        is checked against the normalisation tolerance. If it is smaller than
        this tolerance, the site tensor B is declared right-normalised.
        '''

        B_tensor = self.get_site(pos)

        # index order: s_p | a_p-1 | a_p @ s_p | a'_p-1 | a_p => a_p-1 | a'_p-1
        matrix = tensordot(B_tensor, self._dagger(B_tensor), ((0, 2), (0, 1)))

        # return whether maximum deviation from identity matrix
	# is below normalisation tolerance
        if type(matrix) is np.ndarray:

            # for dense numpy tensors
            dim_matrix = np.shape(matrix)[0]
            matrix = matrix - np.identity(dim_matrix)
            return np.max(np.abs(matrix)) < self._normalisation_tolerance

        elif self._useReducedTensors and type(matrix) is rt.reducedTensor:

            # for sparse rt.reducedTensor tensors
            # charge vectors are equal for both dimensions

            max_err = 0
            for sector in matrix.sectors.keys():

                submatrix = matrix.sectors[sector]
                dim_matrix = np.shape(submatrix)[0]
                submatrix = submatrix - np.identity(dim_matrix)
                max_err = max( max_err , np.max(np.abs(submatrix)) )

            return max_err < self._normalisation_tolerance

        else:
            raise TypeError('The site tensor at position {} '.format(pos)+
                            'is neither an np.ndarray or rt.reducedTensor.')

    def set_twosite_tensor(self,pos_left,two_site_tensor,normalization,
                           D=None,max_truncation_error=None,return_USV=False):
        '''
        Takes a site tensor describing two sites, calculates the individual
        site tensors and sets them to positions pos_left and pos_left+1. The
        two-site tensor must be given in the shape (d,d,D_{left},D_{right}),
        where D_{left} and D_{right} are the respective local bond dimensions
        of the bordering site tensors. If reduced tensors are used, the charge
        vectors must match the respective charge vectors of the bordering site
        tensors.

        This function takes the following arguments:

        pos_left             : The position the left site tensor shall have in
                               the MPS
        two_site_tensor      : The two-site tensor to be decomposed into two
                               one-site tensors. The index order must be:
                               s_p | s_p+1 | a_p-1 | a_p
        normalisation        : defines whether the left site tensor is to be
                               left-normalised ('left' or 'l') or the right
                               site tensor is to be right-normalised ('right'
                               or 'r'). May also be set to 'none' or 'n'
                               causing the square root of the singular values
                               to be multiplied into both U and V. May also be
                               set to 'both' or 'b' to discard the singular
                               values causing the two-site tensor to change but
                               the resulting one-site tensors to be both
                               normalised.
        D                    : The local bond dimension to be used. If set to
                               'None', the default value of the MPS will be
                               used instead.
        max_truncation_error : The maximum trunction error to be allowed. Is
                               overruled by D if necessary.
        return_USV           : Decides, whether the tensors resulting from the
                               singular value decomposition should be returned.

        This function returns either the list [D_here,truncation_error] if
        'return_USV' is set to False and [D_here,truncation_error,U,S,V]
        otherwise. The local bond dimension of the newly created bond is given
        by 'D_here', whereas 'truncation_error' is the truncation error that
        resulted in compressing this bond to that dimension. U,S,V are the
        matrices resulting from the SVD.
        '''

        # perform consistency checks
        self._check_if_array(two_site_tensor)
        self._check_if_tensor_has_right_number_of_dimensions(two_site_tensor,4)

        D_left  = np.shape(two_site_tensor)[2]
        D_right = np.shape(two_site_tensor)[3]

        pos2 = (0 if (pos_left == self.get_length() - 1 or pos_left == -1)
                else pos_left+1)

        #check if local state space matches for the left site tensor
        d_left = np.shape(two_site_tensor)[0]
        if d_left != self.get_local_state_space_dimension():
            raise ValueError('Local state space for left tensor is '
                             '{} but must be '.format(d_left)+
                          '{}.'.format(self.get_local_state_space_dimension()))

        #check if local state space matches for the right site tensor
        d_right = np.shape(two_site_tensor)[1]
        if d_right != self.get_local_state_space_dimension():
            raise ValueError('Local state space for right tensor is '
                             '{} but must be '.format(d_right)+
                          '{}.'.format(self.get_local_state_space_dimension()))

        #check if D_left has the right size
        D_left_current = np.shape(self.get_site(pos_left))[1]
        if D_left != D_left_current:
            raise ValueError('Length of left local bond dimension must be '
                             '{} but is {}.'.format(D_left_current,D_left))

        #check if D_right has the right size
        D_right_current = np.shape(self.get_site(pos2))[2]
        if D_right != D_right_current:
            raise ValueError('Length of right local bond dimension must be '
                             '{} but is {}.'.format(D_right_current,D_right))

        #use only positive indices
        if pos_left < 0:
            pos_left = pos_left + self.get_length() 
            pos2 = pos_left+1 if pos_left < self.get_length() - 1 else 0

        # Additional checks for reduced tensors
        if self._useReducedTensors:

            # charge vectors
            q0 = [-1,1]
            q1 = self.get_site(pos_left).q_vectors[1] if pos_left > 0 else [0]
            q2 = (self.get_site(pos2).q_vectors[2]
                  if pos_left < self.get_length() else [0])
            _check_if_reduced_tensor_has_q_vectors(two_site_tensor,
                                                   [q0,q0,q1,q2])

            # charge signs
            if list(two_site_tensor.q_signs) != [1,1,-1,1]:
                raise TypeError('Tensor has wrong charge vectors.')

        D_here = D if D is not None else self.get_local_bond_dimension()

        #index order: s_p | s_p+1 | a_l-1 | a_l+1
        #          => s_p | a_l-1 | s_p+1 | a_l+1
        two_site_tensor = two_site_tensor.swapaxes(1,2)

        # shape into matrix for SVD
        # index order:  s_p | a_p-1  |  s_p+1 | a_p+1
        #           => (s_p | a_p-1) | (s_p+1 | a_p+1)
        if self._useReducedTensors:
            rt.combine_axes_for_reduced_tensor(two_site_tensor,+1,2,2)
            rt.combine_axes_for_reduced_tensor(two_site_tensor,-1,0,2)
            matrix = two_site_tensor

        else:
            matrix = two_site_tensor.reshape([d_left * D_left,
                                              d_right * D_right])

        # perform SVD
        # U index order: (s_p | a_p-1) | a_p
        # V index order: a_p | (s_p+1 | a_p+1)
        # S index order: a_p
        if self._useReducedTensors:
            U,S,V,truncation_error = rt.SVD_for_reduced_matrix(matrix,
                            D=D_here,max_truncation_error=max_truncation_error,
                            normalize=normalization)
        else:

            if normalization in ['left','l']:
                return_ = 'U|SV'
                normalize = 'S'
            elif normalization in ['right','r']:
                return_ = 'US|V'
                normalize = 'S'
            elif normalization in ['none','n']:
                return_ = 'Us|sV'
                normalize = 'S'
            elif normalization in ['both','b']:
                return_ = 'U|V|S'
                normalize = 'V'
            else:
                raise ValueError("'normalization' must either be 'left', 'l',"
                                 " 'right','r','none', 'n', 'both' or 'b'.")

            X = perform_np_SVD(matrix, D=D_here,
                               max_truncation_error=max_truncation_error,
                               normalize=normalize, return_=return_)

            D_here, truncation_error, U, V = X[:4]
            if len(X) == 5:
                S = X[4]

        # reshape U and V into tensors
        # index order: (s_p | a_p-1) | a_p    => s_p | a_p-1 | a_p
        # index order:  a_p | (s_p+1 | a_p+1) => a_p | s_p+1 | a_p+1
        if self._useReducedTensors:
            rt.split_axis_for_reduced_tensor(U,0)
            rt.split_axis_for_reduced_tensor(V,1)
        else:
            U=U.reshape(self.get_local_state_space_dimension(),D_left,D_here)
            V=V.reshape(D_here,self.get_local_state_space_dimension(),D_right)

        # index order: a_p | s_p+1 | a_p+1 => s_p+1 | a_p | a_p+1
        V = V.swapaxes(0,1)

        # set the new site tensors
        self._MPS[pos_left] = U
        self._MPS[pos2]     = V

        # change A- and B-boundary if necessary
        if self.get_boundary_condition() == 'OBC':
            if normalization in ['left','l']:
                if self.get_left_normalized_border() >= pos_left:
                    self._A_border = pos_left+1
                if self.get_right_normalized_border() < pos_left+2:
                    self._B_border = pos_left+2

            elif normalization in ['right','r']:
                if self.get_left_normalized_border() > pos_left:
                    self._A_border = pos_left
                if self.get_right_normalized_border() <= pos_left+2:
                    self._B_border = pos_left+1

            elif normalization in ['none','n']:
                if self.get_left_normalized_border() > pos_left:
                    self._A_border = pos_left
                if self.get_right_normalized_border() < pos_left+2:
                    self._B_border = pos_left+2

            elif normalization in ['both','b']:
                if self.get_left_normalized_border() > pos_left:
                    self._A_border = pos_left + 1
                if self.get_right_normalized_border() < pos_left+2:
                    self._B_border = pos_left + 1

        # return results
        if return_USV:
            return [D_here,truncation_error,U,S,V]
        else:
            return [D_here,truncation_error]

    def set_twosite_tensor_from_right(self,pos_right_from_right,
                                      two_site_tensor,normalization,D=None,
                                      max_truncation_error=None):
        '''
        Same as 'set_twosite_tensor' but counts from right.
        'pos_right_from_right' is the position of the resulting right site
        tensor counted from the right. The index order of the tensor
        'two_site_tensor' must be: s_p-1 | s_p | a_p-2 | a_p (with p counting
        from left).
        '''

        pos_left = - pos_right_from_right - 2
        return self.set_twosite_tensor(pos_left,two_site_tensor,normalization,
                                       D,max_truncation_error)

    def get_twosite_tensor(self,pos_left):
        '''
        Get the combined site tensor of rank 4, which describes site pos_left
        and pos_left + 1 simultaneously. The order of dimensions is
        (d_{left},d_{right},D_{left},D_{right}). The index order of the given
        tensor is: s_p | s_p+1 | a_p-1 | a_p+1.
        '''

        pos2 = (0 if (pos_left == self.get_length() - 1 or pos_left == -1)
                else pos_left+1)

        # index order: s_p | a_p-1 | a_p @ s_p+1 | a_p | a_p+1
        #           => s_p | a_p-1 | s_p+1 | a_p+1
        combined_site_tensor = tensordot(self.get_site(pos_left),
                                         self.get_site(pos2),[[2],[1]])

        # index order: s_p | a_p-1 | s_p+1 | a_p+1
        #           => s_p | s_p+1 | a_p-1 | a_p+1
        combined_site_tensor = combined_site_tensor.swapaxes(1,2)
        return combined_site_tensor

    def get_twosite_tensor_from_right(self,pos_right_from_right):
        '''
        Same as 'get_twosite_tensor' but counts from right.
        'pos_right_from_right' is the position of the resulting right site
        tensor counted from the right. The index order of the given tensor is:
        s_p-1 | s_p | a_p-2 | a_p (with p counting from left).
        '''

        pos_left = (- pos_right_from_right - 2 if
                    (pos_right_from_right < self.get_length()-1) else -1)
        return self.get_twosite_tensor(pos_left)

    def set_several_site_tensors(self,pos_start,list_of_tensors):
        '''
        This function allows to set multiple site tensors simultaneously. The
        dimensionalities inside the tensor-string has to match as well as the
        connection to the remaining part of the current MPS. The same is true
        for the charge vectors if reduced tensors are employed. The index
        orders of the supplied site tensors must be: s_p | a_p-1 | a_p and
        s_p+1 | a_p | a_p+1 and s_p+2 | a_p+1 | a_p+2 ...
        '''

        # perform consistency checks
        
        # Raise NotImplementedError for PBC
        if self.get_boundary_condition() != 'OBC':
            raise NotImplementedError('Setting several site tensors are not '
                                      'implemented for non-open boundary '
                                      'conditions.')

        # check if position of first tensor is valid
        self._check_if_position_is_valid(pos_start)

        # check if number of tensors is valid or if we overshoot
        number_of_tensors = len(list_of_tensors)
        if pos_start + number_of_tensors > self.get_length():
            raise ValueError('Cannot set {} tensors'.format(number_of_tensors)+
                             '. Only {}'.format(self.get_length()-pos_start)+
                             ' are left right of position '
                             '{}.'.format(pos_start))

        # check if the tensors are numpy arrays
        # and if they have matching dimensions
        if pos_start == 0:
            D_right_from_tensor_left = 1
        else:
            D_right_from_tensor_left = np.shape(self._MPS[pos_start-1])[2]

        for i,tensor in enumerate(list_of_tensors):
            self._check_if_array(tensor)
            self._check_if_tensor_has_right_number_of_dimensions(tensor,3)

            if np.shape(tensor)[1] == D_right_from_tensor_left:

                D_right_from_tensor_left = np.shape(tensor)[2]

            else:
                raise ValueError("Dimensions of the site tensors don't match "
                                 "at position {} in the ".format(i)+
                                 "list of tensors.")

        # check last connection between last tensor in the list
        # and environment to the right
        if pos_start + number_of_tensors == self.get_length():
            # no environment to the right
            if np.shape(list_of_tensors[-1])[2] != 1:
                raise ValueError('Dimensions of the site tensors '
                                 'do not match at the last position '
                                 'in the list of tensors.')
        else:
            if (np.shape(list_of_tensors[-1])[2] !=
                np.shape(self._MPS[pos_start + number_of_tensors])[1]):
                raise ValueError('Dimensions of the site tensors '
                                 'do not match at the last position '
                                 'in the list of tensors.')

        # set tensors (did checks separately to prevent dataloss
        # if something went wrong)
        for i,tensor in enumerate(list_of_tensors):

            # set tensor
            self._MPS[pos_start + i] = tensor

            # check A boundary
            if self.get_left_normalized_border() > pos_start + i:
                # new tensor in the middle of A regime
                if not self.check_if_site_left_normalized(pos_start + i):
                    self._A_border = pos_start + i

            elif self.get_left_normalized_border() == pos_start + i:
                # new tensor at the border of A regime
                if self.check_if_site_left_normalized(pos_start + i):
                    self._A_border = pos_start + i + 1

            # check B boundary
            if self.get_right_normalized_border() < pos_start + i + 1:
                # new tensor in the middle of B regime
                if not self.check_if_site_right_normalized(pos_start + i):
                    self._B_border = pos_start + i + 1

            elif self.get_right_normalized_border() == pos_start + i + 1:
                # new tensor at the border of A regime
                if self.check_if_site_right_normalized(pos_start + i):
                    self._B_border = pos_start + i

    def move_left_normalized_boundary_to_the_right(self,steps=1):
        '''
        Move the boundary of left normalized site tensors to the right by
        'steps' steps. This results in the next 'steps' site tensors right of
        the boundary to become left-normalized. If 'steps' is set to -1, all
        sites will be left normalized. Using this method, there will usually be
        one unnormalized site tensor between the left- and right-normalized
        regime.
        '''

        # perform consistency checks
        self._check_for_open_boundary_conditions()

        if steps <= -1:
            steps = (self.get_length() - self.get_left_normalized_border()
                     + steps + 1)

        if self.get_length() - self.get_left_normalized_border() < steps:
            raise ValueError('Cannot left-normalize {} site '.format(steps)+
                             'tensors, only {}'.format(self.get_length()
                                    - self.get_left_normalized_border())+
                             ' non-left-normalized are left.')

        # perform left-normalisation
        for i in range(steps):

            # perform QR decomposition on the site tensor next to the border
            pos_site_tensor = self.get_left_normalized_border()

            tensor_shape = np.shape(self._MPS[pos_site_tensor])
            tensor_size  = np.size(self._MPS[pos_site_tensor])

            # shape site tensor into matrix
            # index order s_p | a_p-1 | a_p => (s_p | a_p-1) | a_p
            if self._useReducedTensors:
                rt.combine_axes_for_reduced_tensor(self._MPS[pos_site_tensor],
                                                   -1,0,2)
                matrix = self._MPS[pos_site_tensor]
            else:
                matrix = (self._MPS[pos_site_tensor].
                          reshape([tensor_shape[0]*tensor_shape[1],
                                   tensor_shape[2]]))

            # perform QR decomposition
            # index order Q: (s_p | a_p-1) | a'_p
            # index order R: a'_p | a_p
            Q, R = perform_QR(matrix)

            if tensor_size != np.size(Q):
                # Q cannot be reshaped into old shape. Nothing to worry, this
                # can happen if the bond dimension near the edges is increasing
                # faster than exponential. Update a_{left} to compensate the
                # shape shift.

                tensor_shape=list(tensor_shape) # cannot change values in tuple
                tensor_shape[2]=int(tensor_shape[2] * np.size(Q) / tensor_size)

            # Set Q as new left-normalized site tensor
            # index order: (s_p | a_p-1) | a'_p => s_p | a_p-1 | a'_p
            if self._useReducedTensors:
                rt.split_axis_for_reduced_tensor(Q,0)
                self._MPS[pos_site_tensor] = Q
            else:
                self._MPS[pos_site_tensor] = Q.reshape(tensor_shape)

            # Absorb R into the tensor to the right.
            if pos_site_tensor == self.get_length() - 1:

                # if normalized last tensor in MPS: absorp R into
                # multiplicative_factor
                if self._useReducedTensors:
                    aux = R.get_element_in_tensor_coords([0,0])
                    self._multiplicative_factor *= aux
                else:
                    self._multiplicative_factor *= R[0,0]

            else:

                # index order: a'_p | a_p @ s_p | a_p | a_p+1
                #           => a'_p | s_p | a_p+1
                self._MPS[pos_site_tensor+1] = tensordot(R,
                                        self._MPS[pos_site_tensor+1],[[1],[1]])
                # index order: a'_p | s_p | a_p+1 => s_p | a'_p | a_p+1
                self._MPS[pos_site_tensor+1] = (self._MPS[pos_site_tensor+1].
                                                swapaxes(0,1))

            #increase value of the border
            self._A_border += 1
            if self._A_border >= self._B_border:
                if pos_site_tensor == self.get_length() - 1:
                    self._B_border = self._A_border
                else:
                    self._B_border = self._A_border + 1

    def move_right_normalized_boundary_to_the_left(self,steps=1):
        '''
        Move the boundary of right normalized site tensors to the left by
        'steps' steps. This results in the next 'steps' site tensors left of
        the boundary to become right-normalized. If 'steps' is set to -1, all
        sites will be right normalized. Using this method, there will always be
        one unnormalized site tensor between the left- and right-normalized
        regime.
        '''

        # perform consistency checks
        self._check_for_open_boundary_conditions()

        if steps <= -1:
            steps = self.get_right_normalized_border() + steps + 1

        if self.get_right_normalized_border() < steps:
            raise ValueError('Cannot left-normalize {} site '.format(steps)+
                             'tensors, only '
                             '{}'.format(self.get_right_normalized_border())+
                             ' non-left-normalized are left.')

        # perform right-normalisation
        for i in range(steps):

            #perform QR decomposition on the site tensor next to the border
            pos_site_tensor = self.get_right_normalized_border() - 1

            tensor_shape = np.shape(self._MPS[pos_site_tensor])
            tensor_size  = np.size( self._MPS[pos_site_tensor])

            # swap axes for QR decomposition (QR <=> RQ)
            # index order: s_p | a_p-1 | a_p => a_p-1 | s_p | a_p
            matrix = self._MPS[pos_site_tensor].swapaxes(0,1)

            # shape site tensor into matrix
            # index order:  a_p-1 | s_p | a_p =>  a_p-1 | (s_p | a_p)
            if self._useReducedTensors:
                rt.combine_axes_for_reduced_tensor(matrix,+1,1,2)
            else:
                matrix = matrix.reshape([tensor_shape[1],
                                         tensor_shape[0]*tensor_shape[2]])

            # perform QR decomposition
            # index order matrix after dagger: (s_p | a_p) | a_p-1
            # index order Q: (s_p | a_p) | a'_p-1
            # index order R: a'_p-1 | a_p-1
            Q, R = perform_QR(self._dagger(matrix))

            # index order: a'_p-1 | (s_p | a_p)
            Q = self._dagger(Q)

            # index order: a_p-1 | a'_p-1
            R = self._dagger(R)

            if tensor_size != np.size(Q):
                # Q cannot be reshaped into old shape. Nothing to worry, this
                # can happen if the bond dimension near the edges is increasing
                # faster than exponential. Update a_{right} to compensate the
                # shape shift.

                tensor_shape=list(tensor_shape) # cannot change values in tuple
                tensor_shape[1]=int(tensor_shape[1] * np.size(Q) / tensor_size)

            #Set Q as new right-normalized site tensor
            # index order: a'_p-1 | (s_p | a_p) => a'_p-1 | s_p | a_p
            if self._useReducedTensors:
                rt.split_axis_for_reduced_tensor(Q,1)
            else:
                Q=Q.reshape([tensor_shape[1],tensor_shape[0],tensor_shape[2]])

            # index order: a'_p-1 | s_p | a_p => s_p | a'_p-1 |  a_p
            self._MPS[pos_site_tensor] = Q.swapaxes(0,1)

            # absorb R into the tensor to the left
            if pos_site_tensor == 0:

                # if normalized last tensor in MPS: absorp R into
                # multiplicative_factor 
                if self._useReducedTensors:
                    aux = R.get_element_in_tensor_coords([0,0])
                    self._multiplicative_factor *= aux
                else:
                    self._multiplicative_factor *= R[0,0]

            else:

                # index order: s_p-1 | a_p-2 | a_p-1 @ a_p-1 | a'_p-1
                #           => s_p-1 | a_p-2 | a'_p-1
                self._MPS[pos_site_tensor-1] = tensordot(
                    self._MPS[pos_site_tensor-1],R,[[2],[0]])

            #decrease value of the border
            self._B_border -= 1
            if self._B_border <= self._A_border:
                if pos_site_tensor == 0:
                    self._A_border = self._B_border
                else:
                    self._A_border = self._B_border - 1

    def make_MPS_left_canonical(self):
        '''
        Makes the MPS left-canonical, i.e. every site tensor left-normalized.
        '''

        self._check_for_open_boundary_conditions()
        self.move_left_normalized_boundary_to_the_right(-1)

    def make_MPS_right_canonical(self):
        '''
        Makes the MPS right-canonical, i.e. every site tensor right-normalized.
        '''

        self._check_for_open_boundary_conditions()
        self.move_right_normalized_boundary_to_the_left(-1)

    def insert_site_tensor(self,pos,site_tensor,different_shape=False):
        '''
        Inserts a site tensor at position 'pos'. All site tensors right of it
        get assigned a new index (the old one + 1). The new site tensor has to
        match the local bond dimensions of the neighbouring sites. If reduced
        tensors are used, charge vectors have to match as well. The index order
        of the given site tensor must be: s_p | a_p-1 | a_p, where p refers to
        the newly created position. For different_shape=False:
        len(a_p-1) = len(a_p).
        '''

        # perform consistency checks

        # not yet implemented for PBC
        if self.get_boundary_condition() != 'OBC':
            raise NotImplementedError('Inserting site tensors is not '
                                      'implemented for non-open boundary '
                                      'conditions.')

        # check if position is valid
        if pos > self.get_length() or pos < -self.get_length() - 1:
            raise ValueError('The new site tensor cannot be inserted at '
                             'position {} as it does not exist.'.format(pos))

        self._check_if_array(site_tensor)
        self._check_if_tensor_has_right_number_of_dimensions(site_tensor,3)

        tensor_shape = np.shape(site_tensor)

        d = self.get_local_state_space_dimension()
        D_left  =  np.shape(self.get_site(pos-1))[2] if pos != 0 else 1
        D_right = (np.shape(self.get_site(pos))[1]
                   if pos != self.get_length() else 1)

        if tensor_shape != (d,D_left,D_right) and different_shape == False:
            raise ValueError('The shape of the new site does not match the '
                             'MPS. The shape shoud be {} '.format((d,D_left,
                                                                   D_right))+
                             'but is {}.'.format(tensor_shape))

        # from here: positive indices
        if pos < 0:
            pos = pos + self.get_length()
            
        # Additional checks for reduced tensors
        if self._useReducedTensors:

            # charge vectors
            q0 = [-1,1]
            q1 =  self.get_site(pos-1).q_vectors[2] if pos > 0 else [0]
            q2 = (self.get_site(pos).q_vectors[1]
                  if pos < self.get_length() else [0])
            _check_if_reduced_tensor_has_q_vectors(site_tensor,[q0,q1,q2])

            # charge signs
            if list(site_tensor.q_signs) != [1,-1,1]:
                raise TypeError('Tensor has wrong charge vectors.')

        # insert tensor and change properties of MPS
        self._MPS.insert(pos,site_tensor)
        self._L += 1
        self._B_border += 1

        # check A boundary and B-boundary

        # A boundary
        if self.get_left_normalized_border() == pos:
            if self.check_if_site_left_normalized(pos):
                self._A_border += 1
        elif self.get_left_normalized_border() > pos:
            if not self.check_if_site_left_normalized(pos):
                self._A_border = pos

        # B boundary
        if self.get_right_normalized_border() == pos+1:
            if self.check_if_site_right_normalized(pos):
                self._B_border -= 1
        elif self.get_right_normalized_border() < pos+1:
            if not self.check_if_site_right_normalized(pos):
                self._B_border = pos+1

    def insert_twosite_tensor(self,pos,two_site_tensor,normalization,D=None,
                              max_truncation_error=None,return_USV=False):
        '''
        Takes a site tensor describing two sites, calculates the individual
        site tensors and inserts them to positions pos and pos+1. All site
        tensors right of them get assigned new indices (old + 2). The two-site
        tensor must be given in the shape (d,d,D_{left},D_{right}), where
        D_{left} and D_{right} are the respective local bond dimensions of the
        bordering site tensors. If reduced tensors are used, the charge vectors
        must match the respective charge vectors of the bordering site tensors.

        This function takes the following arguments:

	pos_left             : The position the left site tensor shall have in
			       the MPS
	two_site_tensor      : The two-site tensor to be decomposed into two
			       one-site tensors. Index order must be:
                               s_p | s_p+1 | a_p-1 | a_p+1, where p refers to
                               the newly created position. For
                               different_shape=False: len(a_p) = len(a_p+1).
        normalisation        : defines whether the left site tensor is to be
                               left-normalised ('left' or 'l') or the right
                               site tensor is to be right-normalised ('right'
                               or 'r') May also be set to 'none' or 'n' causing
                               the square root of the singular values to be
                               multiplied into both U and V. May also be set to
                               'both' or 'b' to discard the singular values
                               causing the two-site tensor to change but the
                               resulting one-site tensors to be both
                               normalised.
        D                    : The local bond dimension to be used. If set to
                               'None', the default value of the MPS will be
                               used instead.
        max_truncation_error : The maximum trunction error to be allowed. Is
                               overruled by D if necessary.
        return_USV           : Decides, whether the matrices resulting from the
                               singular value decomposition should be returned.

        This function returns either the list [D_here,truncation_error] if
        'return_USV' is set to False and [D_here,truncation_error,U,S,V]
        otherwise. The local bond dimension of the newly created bond is given
        by 'D_here', whereas 'truncation_error' is the truncation error that
        resulted in compressing this bond to that dimension. U,S,V are the
        matrices resulting from the SVD.
        '''

        # perform consistency checks

        # Not yet implemented for PBC
        if self.get_boundary_condition() != 'OBC':
            raise NotImplementedError('Inserting site tensors is not '
                                      'implemented for non-open boundary '
                                      'conditions.')

        self._check_if_array(two_site_tensor)
        self._check_if_tensor_has_right_number_of_dimensions(two_site_tensor,4)

        D_left  = np.shape(two_site_tensor)[2]
        D_right = np.shape(two_site_tensor)[3]

        d = self.get_local_state_space_dimension()

        # check if local state space matches for the left site tensor
        d_left = np.shape(two_site_tensor)[0]
        if d_left != d:
            raise ValueError('Local state space for left tensor is '
                             '{} but must be {}.'.format(d_left,
                                       self.get_local_state_space_dimension()))

        # check if local state space matches for the right site tensor
        d_right = np.shape(two_site_tensor)[1]
        if d_right != d:
            raise ValueError('Local state space for right tensor is '
                             '{} but must be {}.'.format(d_right,
                                       self.get_local_state_space_dimension()))

        # check if D_left has the right size
        D_left_current = np.shape(self.get_site(pos-1))[2] if pos != 0 else 1
        if D_left != D_left_current:
            raise ValueError('Length of left local bond dimension must be '
                             '{} but is {}.'.format(D_left_current,D_left))

        # check if D_right has the right size
        D_right_current = (np.shape(self.get_site(pos))[1]
                           if pos != self.get_length() else 1)
        if D_right != D_right_current:
            raise ValueError('Length of right local bond dimension must be '
                             '{} but is {}.'.format(D_right_current,D_right))

        #use only positiv indices
        if pos < 0:
            pos = pos + self.get_length()

        # create dummy site tensors at the requested sites
        if self._useReducedTensors:
            q0 = self.get_site(pos-1).q_vectors[2]
            q1 = self.get_site(pos  ).q_vectors[1]

            dummy1 = rt.reducedTensor({},[(-1,1),q0,(0,)],[1,-1,1],
                                      sectors_given=True)
            dummy2 = rt.reducedTensor({},[(-1,1),(0,),q1],[1,-1,1],
                                      sectors_given=True)

            self._MPS.insert(pos,  dummy1)
            self._MPS.insert(pos+1,dummy2)
        else:
            self._MPS.insert(pos,np.empty([d,D_left_current,1]))
            self._MPS.insert(pos+1,np.empty([d,1,D_right_current]))

        # change properties of the MPS
        self._L += 2
        self._B_border += 2

        # use set_twosite_tensor to do all the hard work
        return self.set_twosite_tensor(pos_left=pos,
                                       two_site_tensor=two_site_tensor,
                                       normalization=normalization,D=D,
                                     max_truncation_error=max_truncation_error,
                                       return_USV=return_USV)

    ###########################################
    # functions concerning the MPS as a whole #
    ###########################################

    def multiply_scalar(self,scalar):
        '''
        Multiplies the MPS by a scalar. The site tensors remain untouched. The
        scalar will be multiplied to a special variable which, together with
        the site tensors, form the MPS. Setting this variable to zero is
        forbidden as the zero-state is no sensible quantum state.
        '''

        if scalar == 0:
            raise ValueError('The scalar may not be set to zero as this makes '
                             'no sense quantum mechanically.')

        self._multiplicative_factor *= scalar

    def set_multiplication_constant(self,value):
        '''
        Sets the multiplication variable directly. It is prohibited to set it
        to zero.
        '''

        if value == 0:
            raise ValueError('The multiplication variable may not be set to ze'
                             'ro as this makes no sense quantum mechanically.')

        self._multiplicative_factor = value

    def get_multiplication_constant(self):
        '''
        Returns the multiplication constant.
        '''

        return self._multiplicative_factor

    def add(self,MPS,factor=1.,ignore_multiplicative_factor=True):
        '''
        Add a given MPS to the MPS stored in this instance. The MPS is rejected
        if the parameters d and L are not identical. A factor can be given
        which will be applied to the MPS in the fashion |c> = |a> + factor |b>
        with |a> being this MPS, |b> being the supplied second MPS and |c>
        being this MPS after the operation is completed. The local bond
        dimension will become the sum of the local bond dimensions from both
        MPSs.
        '''

        # perform consistency checks
        self._check_if_external_MPS_is_compatible(MPS)

        openBC = (True if self.get_boundary_condition() == 'OBC'
                  and MPS.get_boundary_condition() == 'OBC' else False)

        d1 = self.get_local_state_space_dimension()
        L1 = self.get_length()

        if self._useReducedTensors != MPS._useReducedTensors:
            raise NotImplementedError("Both MPS's must be either dense or use"
                                      " reducedTensors.")

        # perform addition ...
        if self._useReducedTensors:

            # ... for reduced tensors

            # Update every site
            for site in range(L1):

                tensor1 = self.get_site(site)
                tensor2 =  MPS.get_site(site)

                if tensor1.q_signs != tensor2.q_signs:
                    raise NotImplementedError('The charge signs of the two '
                        'site tensors at position {} are not '.format(site)+
                        'equal. This is not a big issue but the correct '
                        'behavior for this situation is not yet implemented. '
                        '(Flip the respective charge sign and multiply the '
                        'corresponding charge vector with -1 to make this '
                        'error go away.')

                # make modifications to first site only
                if site == 0:
                    tensor2 = rt.multiply_by_scalar(tensor2,factor)

                    if not ignore_multiplicative_factor:
                        tensor1 = rt.multiply_by_scalar(tensor1,
                                            self.get_multiplication_constant())
                        tensor2 = rt.multiply_by_scalar(tensor2,
                                             MPS.get_multiplication_constant())
                        self.set_multiplication_constant(1.0)

                tensor1_old_q_vectors = copy.deepcopy(tensor1.q_vectors)

                # update charge vectors and charge signs
                if site == 0 and openBC:
                    tensor1.q_vectors[2].extend(tensor2.q_vectors[2])
                elif site == L1 - 1 and openBC:
                    tensor1.q_vectors[1].extend(tensor2.q_vectors[1])
                else:
                    tensor1.q_vectors[1].extend(tensor2.q_vectors[1])
                    tensor1.q_vectors[2].extend(tensor2.q_vectors[2])

                tensor1.shape = tuple(len(tensor1.q_vectors[i])
                                      for i in range(tensor1.ndim))
                tensor1.size  = np.prod(tensor1.shape)

                # check for omitted sectors in tensor2
                # or sectors that have to be enlarged
                for sector in tensor1.sectors:
                    if sector not in tensor2.sectors:
                        if (sector[1] in tensor2.q_vectors[1] or
                            sector[2] in tensor2.q_vectors[2]):

                            sector1 = tensor1.sectors[sector]

                            Ds2_left  = tensor2.q_vectors[1].count(sector[1])
                            Ds2_right = tensor2.q_vectors[2].count(sector[2])

                            Ds_left = (1 if site == 0 and openBC
                                       else np.shape(sector1)[1] + Ds2_left)
                            Ds_right = (1 if site == L1 - 1 and openBC
                                        else np.shape(sector1)[2] + Ds2_right)

                            new_sector = np.zeros([1,Ds_left,Ds_right],
                                                  dtype=sector1.dtype.type)

                            if site == 0 and openBC:
                                new_sector[0,:,:Ds2_right] = sector1[0]
                            elif site == L1-1 and openBC:
                                new_sector[0,:Ds2_left] = sector1[0]
                            else:
                                new_sector[0,:Ds2_left,:Ds2_right] = sector1[0]

                            tensor1.sectors[sector] = new_sector


                # update charge sectors
                for sector in tensor2.sectors:

                    if sector in tensor1.sectors:
                        # combine both sectors

                        sector1 = tensor1.sectors[sector]
                        sector2 = tensor2.sectors[sector]

                        Ds_left = (1 if site == 0 and openBC else
                                   np.shape(sector1)[1] + np.shape(sector2)[1])
                        Ds_right = (1 if site == L1 - 1 and openBC else
                                    np.shape(sector1)[2]+np.shape(sector2)[2])

                        new_type = _give_convenient_numpy_float_complex_type(
                                         sector1.dtype.type,sector2.dtype.type)
                        new_sector = np.empty([1,Ds_left,Ds_right],
                                              dtype=new_type)

                        if site == 0 and openBC:
                            new_sector[0] = np.concatenate((sector1[0],
                                                            sector2[0]),axis=1)
                        elif site == L1-1 and openBC:
                            new_sector[0] = np.concatenate((sector1[0],
                                                            sector2[0]),axis=0)
                        else:
                            new_sector[0] = sp.linalg.block_diag(sector1[0],
                                                                 sector2[0])

                        tensor1.sectors[sector] = new_sector

                    else:
                        # add sector to tensor1

                        sector2 = tensor2.sectors[sector]

                        if (sector[1] in tensor1_old_q_vectors[1]
                            or sector[2] in tensor1_old_q_vectors[2]):
                            # sector present in tensor1 but was omitted
                            # or sector has to be enlarged

                            Ds1_left =tensor1_old_q_vectors[1].count(sector[1])
                            Ds1_right=tensor1_old_q_vectors[2].count(sector[2])

                            Ds_left = (1 if site == 0 and openBC else
                                       Ds1_left  + np.shape(sector2)[1])
                            Ds_right = (1 if site == L1 - 1 and openBC else
                                        Ds1_right + np.shape(sector2)[2])

                            new_sector = np.zeros([1,Ds_left,Ds_right],
                                                  dtype=sector2.dtype.type)

                            if site == 0 and openBC:
                                new_sector[0,:,Ds1_right:] = sector2[0]
                            elif site == L1-1 and openBC:
                                new_sector[0,Ds1_left:] = sector2[0]
                            else:
                                new_sector[0,Ds1_left:,Ds1_right:] = sector2[0]

                            tensor1.sectors[sector] = new_sector

                        else:
                            tensor1.sectors[sector] = tensor2.sectors[sector]

                # update tensor filling and filling density
                tensor1.filling = (np.sum([np.size(tensor1.sectors[sector])
                                           for sector in tensor1.sectors]))
                tensor1.density = tensor1.filling / tensor1.size

        else:

            # ... for dense tensors

            #adding all the matrices for all sites together: C = [[A,0],[0,B]]
            for site in range(L1):
                tensor1 = self.get_site(site)
                tensor2 =  MPS.get_site(site)

                # Evaluate local bond dimension and take factors into account
                if site == 0:
                    D_left = (1 if openBC else
                              np.shape(tensor1)[1] + np.shape(tensor2)[1])
                    tensor2 = factor*tensor2

                    if not ignore_multiplicative_factor:
                        tensor1 = self.get_multiplication_constant() * tensor1
                        tensor2 =  MPS.get_multiplication_constant() * tensor2
                        self.set_multiplication_constant(1.0)

                else:
                    D_left  = np.shape(tensor1)[1] + np.shape(tensor2)[1]

                if openBC and site == L1 - 1:
                    D_right = 1
                else:
                    D_right = np.shape(tensor1)[2] + np.shape(tensor2)[2]

                # create new site tensor
                new_type = _give_convenient_numpy_float_complex_type(
                                         tensor1.dtype.type,tensor2.dtype.type)
                new_tensor = np.empty([d1,D_left,D_right],dtype=new_type)
                for sigma in range(d1):

                    if openBC and site == 0:
                        new_tensor[sigma] = np.concatenate((tensor1[sigma],
                                                            tensor2[sigma]),
                                                           axis=1)
                    elif openBC and site == L1 - 1:
                        new_tensor[sigma] = np.concatenate((tensor1[sigma],
                                                            tensor2[sigma]),
                                                           axis=0)
                    else:
                        new_tensor[sigma]=sp.linalg.block_diag(tensor1[sigma],
                                                               tensor2[sigma])

                self._MPS[site] = new_tensor

        # set properties of the MPS
        self._D = (self.get_local_bond_dimension()
                   + MPS.get_local_bond_dimension())
        self._A_border = 0
        self._B_border = self._L

    def normalize(self):
        '''
        Normalizes the MPS by dividing the first site by the norm of the MPS.
        '''

        norm = self.norm()
        tensor = self.get_site(0)

        if self._useReducedTensors:
            tensor  = rt.multiply_by_scalar(tensor,1/norm)
        else:
            tensor /= norm

        self.set_site(0,tensor)

    # matrix elements on all sites

    def norm(self, ignore_multiplicative_factor=True):
        '''
        Calculates the norm of the MPS. If ignore_multiplicative_factor=False,
        the multiplicative factor of the MPS will be taken into account,
        otherwise the norm will be calculated from the site tensors only.
        '''

        # perform first contraction thus creating the auxiliary tensor 'norm'
        # index order: s_1 | a'_2 | a'_1 @ s_1 | a_1 | a_2
        #           => a'_2 | a'_1 | a_1 | a_2
        norm = tensordot(self._dagger(self.get_site(0)),self.get_site(0),
                         axes=([0],[0]))

        # loop over rest of the tensor network
        for site in range(1,self.get_length()):

            # index order: a'_p-1 | a'_0 | a_0 | a_p-1 @ s_p | a_p-1 | a_p
            #           => a'_p-1 | a'_0 | a_0 | s_p | a_p
            norm = tensordot(norm,self.get_site(site),axes=([3],[1]))

            # index order: s_p | a'_p | a'_p-1
            #            @ a'_p-1 | a'_0 | a_0 | s_p | a_p
            #           => a'_p | a'_0 | a_0 | a_p
            norm = tensordot(self._dagger(self.get_site(site)),norm,
                             axes=([0,2],[3,0]))

        # index order: a'_L | a'_0 | a_0 | a_L => a_0 | a'_0 | a'_L | a_L
        norm = norm.swapaxes(0,2)

        # index order: a_0 | a'_0 | a'_L | a_L => a_0 | a'_0 | a_L | a'_L
        norm = norm.swapaxes(2,3)

        # index order: a_0 | a'_0 | a_L | a'_L => (a_0 | a'_0) | (a_L | a'_L)
        if self._useReducedTensors:
            rt.combine_axes_for_reduced_tensor(norm,1,2,2)
            rt.combine_axes_for_reduced_tensor(norm,1,0,2)
        else:
            norm = norm.reshape(norm.shape[0]*norm.shape[1],
                                norm.shape[2]*norm.shape[3])

        # calculate trace
        norm = np.sqrt(np.abs(trace(norm)))

        # return norm
        if ignore_multiplicative_factor:
            return norm
        else:
            a = self.get_multiplication_constant()
            return norm * np.conjugate(a)*a

    def overlap(self, MPS, ignore_multiplicative_factor=True,dtype=None,
                exponent_cutback_mode=False):
        '''
        Calculates the overlap between the current MPS and the given MPS. Both
        MPSs must have the same length L and same local state space
        dimension d. If ignore_multiplicative_factor=False, the multiplicative
        constants will be taken into account, otherwise the overlap will be
        calculated solely from the site tensors. By setting
        exponent_cutback_mode=True, it is possible to make use of the exponent
        cutback mode which prevents underflows by returning the overlap as the
        list [O1,O2], where O1 is a float and O2 an integer. The overlap itself
        is then O1 + 2^O2, although it may not be possible to calculate this
        quanitity as doing so may result in an underflow. During the usage of
        the exponent cutback mode no additional rounding errors occur. The
        argument 'dtype' can be used to modify the datatype with which this
        calculation is to be performed. This is helpful to delay an underflow
        if the usage of the exponent cutback mode is not desired. The advantage
        of using dtype=np.float128 is that only one number is returned in
        contrast to the exponent cutback mode.
        '''

        def cut_back_exponent(overlap):
            '''
            Takes an array and calculates the lowest magnitude with respect to
            base 2. Afterwards, all entries are raised by the inverse of that
            magnitude thus causing the lower-most magnitude in the array now to
            be zero. Returns the thus modified array alongside the magnitude
            that was removed.
            '''

            if type(overlap) is rt.reducedTensor:
                lowmag = overlap.lowest_magnitude()
                if np.isneginf(lowmag):
                    return overlap, 0

                exponent = int(np.log2(lowmag))
                overlap  = rt.multiply_by_scalar(overlap,np.power(dtype(2.),
                                                                  -exponent))

            elif type(overlap) is np.ndarray:
                nonzero = overlap[np.nonzero(overlap)]
                if nonzero.size == 0:
                    return overlap,0

                exponent = int(np.log2(np.min(np.abs(nonzero))))
                overlap *= np.power(dtype(2.),-exponent)
            else:
                if overlap == 0.0:
                    return 0.0,0

                exponent = int(np.log2(np.min(np.abs(overlap))))
                overlap *= np.power(dtype(2.),-exponent)
            return overlap, exponent


        removed_exponents = 0
        self._check_if_external_MPS_is_compatible(MPS)

        # perform first contraction thus creating auxiliary tensor 'overlap'
        # index order: s_1 | a_2 | a_1 @ s_1 | a'_1 | a'_2
        #           => a_2 | a_1 | a'_1 | a'_2
        dtype = dtype if dtype is not None else self.get_site(0).dtype.type
        overlap = tensordot(self._dagger(self.get_site(0)).astype(dtype),
                            MPS.get_site(0),axes=([0],[0]))

        if exponent_cutback_mode:
            overlap,exponent = cut_back_exponent(overlap)
            removed_exponents += exponent

        # take multiplicative factors into account if desired
        if not ignore_multiplicative_factor:

            overlap *= (np.conjugate(self.get_multiplication_constant())*
                        MPS.get_multiplication_constant())

            if exponent_cutback_mode:
                overlap,exponent = cut_back_exponent(overlap)
                removed_exponents += exponent

        # loop over rest of the tensor network
        for site in range(1,self.get_length()):

            # index order: a_p-1 | a_0 | a'_0 | a'_p-1 @ s_p | a'_p-1 | a'_p
            #           => a_p-1 | a_0 | a'_0 | s_p | a'_p
            overlap = tensordot(overlap,MPS.get_site(site),axes=([3],[1]))

            # index order: s_p | a_p | a_p-1 @ a_p-1 | a_0 | a'_0 | s_p | a'_p
            #           => a_p | a_0 | a'_0 | a'_p
            overlap = tensordot(self._dagger(self.get_site(site)),overlap,
                                axes=([0,2],[3,0]))

            if exponent_cutback_mode:
                overlap,exponent = cut_back_exponent(overlap)
                removed_exponents += exponent

        # index order: a_L | a_0 | a'_0 | a'_L => a'_0 | a_0 | a_L | a'_L
        overlap = norm.swapaxes(0,2)

	# index order: a'_0 | a_0 | a_L | a'_L => a'_0 | a_0 | a'_L | a_L
        overlap = norm.swapaxes(2,3)

        # index order: (a'_0 | a_0) | (a'_L | a_L)
        if self._useReducedTensors:
            rt.combine_axes_for_reduced_tensor(overlap,1,2,2)
            rt.combine_axes_for_reduced_tensor(overlap,1,0,2)
        else:
            overlap = overlap.reshape(overlap.shape[0]*overlap.shape[1],
                                      overlap.shape[2]*overlap.shape[3])

        # calculate trace
        overlap = trace(overlap)

        # cut back the exponent one last time, if desired
        if exponent_cutback_mode:
            overlap,exponent = cut_back_exponent(overlap)
            removed_exponents += exponent

        # return results
        if exponent_cutback_mode:
            return overlap, removed_exponents
        else:
            return overlap

    def expectation_value(self,operator_string,
                          ignore_multiplicative_factor=True):
        '''
        Calculates the expectation value of the MPS with a string of local
        operators only acting on the individual positions. The variable
        'operator_string' is a list holding the local operators for each
        position in the spin chain. A local operator is a matrix of shape (d,d)
        with d being the local state space dimension at this position. If no
        operation for a given position shall be carried out, set the operator
        at that position to None. The index order of the p-th entry in operator
        string must be: s'_p | s_p.
        '''

        # perform consistency checks
        self._check_if_external_operator_string_is_compatible(operator_string)

        # perform first contraction thus creating the auxiliary tensor 'ev'
        if operator_string[0] is None:

            # index order: s_1 | a'_1 | a'_0 @ s_1 | a_0 | a_1
            #           => a'_1 | a'_0 | a_0 | a_1
            ev = tensordot(self._dagger(self.get_site(0)),self.get_site(0),
                           axes=([0],[0]))

        else:

            # index order: s'_1 | s_1 @ s_1 | a_0 | a_1 => s'_1 | a_0 | a_1
            ev = tensordot(operator_string[0],self.get_site(0),axes=([1],[0]))

            # index order: s'_1 | a'_1 | a'_0 @ s'_1 | a_0 | a_1
            #           => a'_1 | a'_0 | a_0 | a_1
            ev = tensordot(self._dagger(self.get_site(0)),ev,axes=([0],[0]))

        # perform contraction
        for site in range(1,self.get_length()):

            # index order: a'_p-1 | a'_0 | a_0 | a_p-1 @ s_p | a_p-1 | a_p
            #           => a'_p-1 | a'_0 | a_0 | s_p | a_p
            ev = tensordot(ev,self.get_site(site),axes=([3],[1]))

            if operator_string[site] is None:

                # index order: s_p | a'_p | a'_p-1
                #            @ a'_p-1 | a'_0 | a_0 | s_p | a_p
                #           => a'_p | a'_0 | a_0 | a_p
                ev = tensordot(self._dagger(self.get_site(site)),ev,
                               axes=([0,2],[3,0]))
            else:

                # index order: s'_p | s_p @ a'_p-1 | a'_0 | a_0 | s_p | a_p
                #           => s'_p | a'_p-1 | a'_0 | a_0 | a_p
                ev = tensordot(operator_string[site],ev,axes=([1],[3]))

                # index order: s'_p | a'_p| a'_p-1
                #            @ s'_p | a'_p-1 | a'_0 | a_0 | a_p
                #           => a'_p | a'_0 | a_0 | a_p
                ev = tensordot(self._dagger(self.get_site(site)),ev,
                               axes=([0,2],[0,1]))

        # index order: a'_L | a'_0 | a_0 | a_L => a_0 | a'_0 | a'_L | a_L
        ev = ev.swapaxes(0,2)

        # index order: a_0 | a'_0 | a'_L | a_L => a_0 | a'_0 | a_L | a'_L
        ev = ev.swapaxes(2,3)

        # index order: a_0 | a'_0 | a_L | a'_L => (a_0 | a'_0) | (a_L | a'_L)
        if self._useReducedTensors:
            rt.combine_axes_for_reduced_tensor(ev,1,2,2)
            rt.combine_axes_for_reduced_tensor(ev,1,0,2)
        else:
            ev = ev.reshape(ev.shape[0]*ev.shape[1],ev.shape[2]*ev.shape[3])

        # calculate trace
        ev = trace(ev)

        # return results
        if ignore_multiplicative_factor:
            return ev
        else:
            return ev * np.conjugate(a)*a

    def matrix_element(self,operator_string,MPS,
                       ignore_multiplicative_factor=True):
        '''
        Calculates the matrix element with a string of local operators, which
        only act at local positions and another MPS. The variable
        'operator_string' is a list holding the local operators for each
        position in the spin chain. A local operator is a matrix of shape (d,d)
        with d being the local state space dimension at this position. If no
        operation for a given position shall be caried out, set the operator at
        that position to None. The index order of the p-th entry in
        operator_string must be: s'_p | s_p.
        '''

        # perform consistency checks
        self._check_if_external_operator_string_is_compatible(operator_string)
        self._check_if_external_MPS_is_compatible(MPS)

        # perform first contraction thus creating the auxiliary tensor 'mel'
        if operator_string[0] is None:

            # index order: s_1 | a'_1 | a'_0 @ s_1 | a_0 | a_1
            #           => a'_1 | a'_0 | a_0 | a_1
            mel = tensordot(self._dagger(self.get_site(0)),MPS.get_site(0),
                            axes=([0],[0]))
        else:

            # index order: s'_1 | s_1 @ s_1 | a_0 | a_1 => s'_1 | a_0 | a_1
            mel = tensordot(operator_string[0],MPS.get_site(0),axes=([1],[0]))

            # index order: s'_1 | a_1 | a_0 @ s'_1 | a_0 | a_1
            #           => a'_1 | a'_0 | a_0 | a_1
            mel = tensordot(self._dagger(self.get_site(0)),mel,axes=([0],[0]))

        # contract the rest of the network
        for site in range(1,self.get_length()):

            # index order: a'_p-1 | a'_0 | a_0 | a_p-1 @ s_p | a_p-1 | a_p
            #           => a'_p-1 | a'_0 | a_0 | s_p | a_p
            mel = tensordot(mel,MPS.get_site(site),axes=([3],[1]))

            if operator_string[site] is None:

                # index order: s_p | a'_p | a'_p-1
                #            @ a'_p-1 | a'_0 | a_0 | s_p | a_p
                #           => a'_p | a'_0 | a_0 | a_p
                mel = tensordot(self._dagger(self.get_site(site)),mel,
                                axes=([0,2],[3,0]))
            else:

                # index order: s'_p | s_p @ a'_p-1 | a'_0 | a_0 | s_p | a_p
                #           => s'_p | a'_p-1 | a'_0 | a_0 | a_p
                mel = tensordot(operator_string[site],mel,axes=([1],[3]))

                # index order: s'_p | a'_p| a'_p-1
                #            @ s'_p | a'_p-1 | a'_0 | a_0 | a_p
                #           => a'_p | a'_0 | a_0 | a_p
                mel = tensordot(self._dagger(self.get_site(site)),mel,
                                axes=([0,2],[0,1]))

        # index order: a'_L | a'_0 | a_0 | a_L => a_0 | a'_0 | a'_L | a_L
        mel = mel.swapaxes(0,2)

        # index order: a_0 | a'_0 | a'_L | a_L => a_0 | a'_0 | a_L | a'_L
        mel = mel.swapaxes(2,3)

        # index order: a_0 | a'_0 | a_L | a'_L => (a_0 | a'_0) | (a_L | a'_L)
        if self._useReducedTensors:
            rt.combine_axes_for_reduced_tensor(mel,1,2,2)
            rt.combine_axes_for_reduced_tensor(mel,1,0,2)
        else:
            mel = mel.reshape(mel.shape[0]*mel.shape[1],
                              mel.shape[2]*mel.shape[3])

        # calculate trace
        mel = trace(mel)

        # return result
        if ignore_multiplicative_factor:
            return mel
        else:
            return mel * np.conjugate(a)*a

    ###################################################
    # functions for local matrix elements on one site #
    ###################################################

    def local_overlap_on_one_site(self,pos,external_site_tensor):
        '''
        Calculates the overlap of this tensor to a tensor on which all sites
        are identical apart from the site tensor at the given position, which
        is given to this function. MPS must be in mixed-canonical form with
        'pos' between both normalized regimes. The index order of the external
        site tensor must be: s_p | a_p-1l | a_p.
        '''

        # perform consistency checks
        self._check_for_open_boundary_conditions()
        self._check_if_position_is_valid(pos)
        self._check_if_array(external_site_tensor)
        self._check_if_tensor_has_right_number_of_dimensions(
                                                        external_site_tensor,3)

        # get shape of both involved tensors
        shape_internal = np.shape(self.get_site(pos))
        shape_external = np.shape(external_site_tensor)

        # from here: positive indices
        if pos < 0:
            pos = pos + self.get_length()

        # perform more consistency checks
        if not np.array_equal(shape_internal,shape_external):
            raise ValueError('The external site tensor has not the same shape'
                             ' {} as the internal site'.format(shape_external)+
                             ' tensor {}.'.format(shape_internal))

        if not (self._A_border == pos or self._A_border == pos + 1):
            raise ValueError("All sites left of 'pos' must be "
                             "left-normalized, but this is not the case.")

        if not (self._B_border == pos or self._B_border == pos + 1):
            raise ValueError("All sites right of 'pos' must be "
                             "right-normalized, but this is not the case.")

        # index order: s_p | a_p | a_p-1 @ s_p | a_p-1 | a_p => a_p | a_p
        overlap = tensordot(self._dagger(self.get_site(pos)),
                            external_site_tensor,axes=([0,2],[0,1]))

        # calculate trace and return overlap
        overlap = trace(overlap)
        return overlap

    def local_expectation_value_on_one_site(self,pos,operator):
        '''
        Calculates the expectation value of the MPS with the supplied local
        operator solely on position 'pos'. For this to work, the MPS must be in
        mixed canonical form with 'pos' being between both normalized regimes.
        The index order of the array 'operator' must be: s'_p | s_p.
        '''

        # perform consistency checks
        self._check_for_open_boundary_conditions()
        self._check_if_position_is_valid(pos)
        self._check_if_array(operator)
        self._check_if_tensor_has_right_number_of_dimensions(operator,2)

        # from here: positive indices
        if pos < 0:
            pos = pos + self.get_length()

        # perform more consistency checks
        d = self.get_local_state_space_dimension()
        if np.shape(operator) != (d,d):
            dim1,dim2 = np.shape(operator)
            raise ValueError('Local operator must be of shape '
                             '({},{}) but operator at position '.format(d,d)+
                             '{} is of shape ({},{}).'.format(i,dim1,dim2))

        if not (self._A_border == pos or self._A_border == pos + 1):
            raise ValueError("All sites left of 'pos' must be "
                             "left-normalized, but this is not the case.")

        if not (self._B_border == pos or self._B_border == pos + 1):
            raise ValueError("All sites right of 'pos' must be "
                             "right-normalized, but this is not the case.")

        # index order: s'_p | a_p | a_p-1 @ s_p | a_p-1 | a_p
        #           => s'_p | a-p | s_p | a_p
        ev = tensordot(self._dagger(self.get_site(pos)),self.get_site(pos),
                       axes=([2],[1]))

        # index order: s'_p | s_p @ s'_p | a-p | s_p | a_p => a-p | a_p
        ev = tensordot(operator,ev,axes=([0,1],[0,2]))

        # calculate trace and return expectation value
        ev = trace(ev)
        return ev

    def local_expectation_value_on_all_sites(self,operator):
        '''
        Calculates the expectation value of a local operator on every site and
        returns them in a list. This function is essentially a wrapper around
        the function 'local_expectation_value_on_one_site'. The index order of
        the array 'operator' must be s'_p | s_p and the local state space
        dimension of all site tensors in the MPS must be equal.
        '''

        # make MPS right-canonical
        self.make_MPS_right_canonical()

        # sweep through chain and calculate the EV at each site
        EV_list = []

        for pos in range(self.get_length()):
            if pos > 0:
                self.move_left_normalized_boundary_to_the_right()
            EV_list.append(self.local_expectation_value_on_one_site(pos,
                                                                    operator))

        return EV_list

    def local_matrix_element_on_one_site(self,pos,operator,
                                         external_site_tensor):
        '''
        Calculates the matrix element of the MPS with the supplied local
        operator and external_site_tensor solely on position 'pos'. For this to
        work, the MPS must be in mixed canonical form with 'pos' being between
        both normalized regimes. The index order of the array 'operator' must
        be s'_p | s_p and the index order of the array 'external_site_tensor'
        must be s_p | a_p-1 | a_p.
        '''

        # perform consistency checks
        self._check_for_open_boundary_conditions()
        self._check_if_position_is_valid(pos)
        self._check_if_array(operator)
        self._check_if_tensor_has_right_number_of_dimensions(operator,2)

        # from here: positive indices
        if pos < 0:
            pos = pos + self.get_length()

        d = self.get_local_state_space_dimension()
        if np.shape(operator) != (d,d):
            dim1,dim2 = np.shape(operator)
            raise ValueError('Local operators must be of shape '
                             '({},{}) but operator at position '.format(d,d)+
                             '{} is of shape ({},{}).'.format(i,dim1,dim2))

        # check external site tensor
        self._check_if_array(external_site_tensor)
        self._check_if_tensor_has_right_number_of_dimensions(
                                                        external_site_tensor,3)

        shape_internal = np.shape(self.get_site(pos))
        shape_external = np.shape(external_site_tensor)

        if not np.array_equal(shape_internal,shape_external):
            raise ValueError('The external site tensor has not the same shape'
                             ' {} as the internal site'.format(shape_external)+
                             ' tensor {}.'.format(shape_internal))

        # check MPS
        if not (self._A_border == pos or self._A_border == pos + 1):
            raise ValueError("All sites left of 'pos' must be "
                             "left-normalized, but this is not the case.")

        if not (self._B_border == pos or self._B_border == pos + 1):
            raise ValueError("All sites right of 'pos' must be "
                             "right-normalized, but this is not the case.")

        # calculate matrix element

        # index order: s'_p | a_p | a_p-1 @ s_p | a_p-1 | a_p
        mel = tensordot(self._dagger(self.get_site(pos)),external_site_tensor,
                        axes=([2],[1]))

        # index order: s'_p | s_p @ s'_p | a-p | s_p | a_p => a-p | a_p
        mel = tensordot(operator,mel,axes=([0,1],[0,2]))

        # calculate trace and return matrix element
        mel = trace(mel)
        return mel

    ####################################################
    # functions for local matrix elements on two sites #
    ####################################################

    def local_overlap_on_two_sites(self,pos_left,two_site_tensor):
        '''
        Calculates the overlap of this tensor to a tensor on which all sites
        are identical apart from the site tensors at pos_left and pos_left + 1.
        They are given externally to this function in form of a combined site
        tensor of rank 4. MPS must be in mixed-canonical position. The given
        two-site tensor must have the index order: s_p | s_p+1 | a_p-1 | a_p+1.
        '''

        # perform consistency checks
        self._check_for_open_boundary_conditions()
        self._check_if_position_is_valid(pos_left)
        self._check_if_position_is_valid(pos_left+1)
        self._check_if_array(two_site_tensor)
        self._check_if_tensor_has_right_number_of_dimensions(two_site_tensor,4)

        shape_external  = np.shape(two_site_tensor)

        d = self.get_local_state_space_dimension()
        D_left  = np.shape(self.get_site(pos_left))[1]
        D_right = np.shape(self.get_site(pos_left+1))[2]

        # from here: positive indices
        if pos_left < 0:
            pos_left = pos_left + self.get_length()

        shape_internal = (d,d,D_left,D_right)

        if not np.array_equal(shape_internal,shape_external):
            raise ValueError('The external site tensor has not the same shape'
                             ' {} as the internal site'.format(shape_external)+
                             ' tensor {}.'.format(shape_internal))

        if not (self._A_border >= pos_left and self._A_border <= pos_left + 2):
            raise ValueError("All sites left of 'pos' must be "
                             "left-normalized, but this is not the case.")

        if not (self._B_border >= pos_left and self._B_border <= pos_left + 2):
            raise ValueError("All sites right of 'pos'+1 must be "
                             "right-normalized, but this is not the case.")

        # calculate overlap

        # index order: s_p | a_p-1 | a_p @ s_p+1 | a_p | a_p+1
        #           => s_p | a_p-1 | s_p+1 | a_p+1
        internal_two_site_tensor = tensordot(self.get_site(pos_left),
                                             self.get_site(pos_left+1),
                                             axes=([2],[1]))

        # index order: s_p | a_p-1 | s_p+1 | a_p+1
        #           => s_p | s_p+1 | a_p-1 | a_p+1
        internal_two_site_tensor = internal_two_site_tensor.swapaxes(1,2)

        # index order: s_p | s_p+1 | a_p+1 | a_p-1
        #            @ s_p | s_p+1 | a_p-1 | a_p+1
        #           => a_p+1 | a_p+1
        overlap = tensordot(self._dagger(internal_two_site_tensor),
                            two_site_tensor,axes=([0,1,3],[0,1,2]))

        # calculate trace and return overlap
        overlap = trace(overlap)
        return overlap

    def local_matrix_element_on_two_sites(self,pos_left,operator1,operator2,
                                          twosite_tensor):
        '''
        Calculates the expectation value of the MPS with the both supplied
        local operators on positions 'pos_left' and 'pos_left' + 1. For this to
        work, the MPS must be in mixed canonical form.
        index order of operator1: s'_p | s_p
        index order of operator2: s'_p+1 | s_p+1
        index order of twosite_tensor: s'_p | s'_p+1 | a_p-1 | a_p+1
        '''

        # perform consistency checks
        self._check_for_open_boundary_conditions()

        self._check_if_position_is_valid(pos_left)
        self._check_if_position_is_valid(pos_left+1)

        self._check_if_array(operator1)
        self._check_if_array(operator2)
        self._check_if_array(twosite_tensor)

        self._check_if_tensor_has_right_number_of_dimensions(operator1,2)
        self._check_if_tensor_has_right_number_of_dimensions(operator2,2)
        self._check_if_tensor_has_right_number_of_dimensions(twosite_tensor,4)

        d = self.get_local_state_space_dimension()
        D_left  = np.shape(self.get_site(pos_left))[1]
        D_right = np.shape(self.get_site(pos_left+1))[2]

        # from here: postitive indices
        if pos_left < 0:
            pos_left = pos_left + self.get_length()

        shape_internal = (d,d,D_left,D_right)
        shape_external  = np.shape(twosite_tensor)

        if np.shape(operator1) != (d,d):
            dim1,dim2 = np.shape(operator1)
            raise ValueError('Local operator 1 must be of shape '
                             '({},{}) but operator at position '.format(d,d)+
                             '{} is of shape ({},{})'.format(i,dim1,dim2))

        if np.shape(operator2) != (d,d):
            dim1,dim2 = np.shape(operator2)
            raise ValueError('Local operator 2 must be of shape '
                             '({},{}) but operator at position '.format(d,d)+
                             '{} is of shape ({},{})'.format(i,dim1,dim2))

        if not np.array_equal(shape_internal,shape_external):
            raise ValueError('The external site tensor has not the same shape '
                             '{} as the internal site '.format(shape_external)+
                             'tensor {}.'.format(shape_internal))

        if not (self._A_border >= pos_left and self._A_border <= pos_left + 2):
            raise ValueError("All sites left of 'pos' must be "
                             "left-normalized, but this is not the case.")

        if not (self._B_border >= pos_left and self._B_border <= pos_left + 2):
            raise ValueError("All sites right of 'pos'+1 must be "
                             "right-normalized, but this is not the case.")

        # calculate matrix element

        # index order: s_p | a_p-1 | a_p @ s_p+1 | a_p | a_p+1
        #           => s_p | a_p-1 | s_p+1 | a_p+1
        internal_twosite_tensor = tensordot(self.get_site(pos_left),
                                            self.get_site(pos_left+1),
                                            axes=([2],[1]))

        # index order: s_p | a_p-1 | s_p+1 | a_p+1
        #           => s_p | s_p+1 | a_p-1 | a_p+1
        internal_twosite_tensor = internal_twosite_tensor.swapaxes(1,2)

        # index order: s'_p+1 | s_p+1 @ s_p | s_p+1 | a_p+1 | a_p-1
        #           => s'_p+1 | s_p | a_p+1 | a_p-1
        internal_twosite_tensor = tensordot(operator2,
                                         self._dagger(internal_twosite_tensor),
                                            axes=([1],[1]))

        # index order: s'_p | s_p @ s'_p+1 | s_p | a_p+1 | a_p-1
        #           => s'_p | s'_p+1 | a_p+1 | a_p-1
        internal_twosite_tensor = tensordot(operator1, internal_twosite_tensor,
                                            axes=([1],[1]))

        # index order: s'_p | s'_p+1 | a_p+1 | a_p-1
        #            @ s'_p | s'_p+1 | a_p-1 | a_p+1
        #           => a_p+1 | a_p+1
        mel = tensordot(internal_twosite_tensor,twosite_tensor,
                        axes=([0,1,3],[0,1,2]))

        # calculate trace and return matrix element
        mel = trace(mel)
        return mel

    ###########################
    # miscellaneous functions #
    ###########################

    def apply_MPO_to_MPS(self,MPO):
        '''
        Apply a given MPO to this MPS. During this calculation, the local bond
        dimension of the MPS grows by a factor equal to the local bond
        dimension of the supplied MPO.
        '''

        # loop through chain and apply site tensors of MPO to MPS
        for l in range(self._L):

            # index order: s_l | a_l-1 | a_l
            site_tensor = self.get_site(l)

            # index order: b_l-1 | b_l | s'_l | s_l @ s_l | a_l-1 | a_l
            #           => b_l-1 | b_l | s'_l | a_l-1 | a_l
            site_tensor = tensordot(MPO[l],site_tensor,axes=([3],[0]))

            # index order: b_l-1 | b_l | s'_l | a_l-1 | a_l
            #           => s'_l | b_l | b_l-1 | a_l-1 | a_l
            site_tensor = site_tensor.swapaxes(0,2)

            # index order: s'_l | b_l | b_l-1 | a_l-1 | a_l
            #           => s'_l | b_l-1 | b_l | a_l-1 | a_l 
            site_tensor = site_tensor.swapaxes(1,2)

            # index order: s'_l | b_l-1 | b_l | a_l-1 | a_l
            #           => s'_l | b_l-1 | a_l-1 | b_l | a_l
            site_tensor = site_tensor.swapaxes(2,3)

            dimM = np.shape(site_tensor)

            # index order: s'_l |  b_l-1 | a_l-1  |  b_l | a_l
            #           => s'_l | (b_l-1 | a_l-1) | (b_l | a_l)
            if type(site_tensor) is np.ndarray:

                site_tensor = site_tensor.reshape(dimM[0],dimM[1]*dimM[2],
                                                  dimM[3]*dimM[4])

            elif (self._useReducedTensors and
                  type(site_tensor) is rt.reducedTensor):

                rt.combine_axes_for_reduced_tensor(site_tensor,+1,3,2)
                rt.combine_axes_for_reduced_tensor(site_tensor,-1,1,2)
                site_tensor = rt.purify_tensor(site_tensor)

            # set site tensor with disabled consistency checks
            self.set_site(l,site_tensor,different_shape=True)

    def compress_MPS_with_SVD(self,D=None,max_truncation_error=None,
                              continuous_sweep=True,even_bonds=True):
        '''
        Compress the MPS by employing the singular value decomposition on every
        bond between neighbouring site tensors. This is done by first summing
        over this bond, then performing the SVD, followed by cutting away a
        number of singular values and adjusting the sizes of the involved
        matrices accordingly. This function is essentially a fancy wrapper
        around the functions get_twosite_tensor and set_twosite_tensor.

        This function takes a number of arguments:

        D                    : The local bond dimension to which the sites
                               should be cut down to. If D=None, the default
                               value of the MPS will be used instead.
        max_truncation_error : The largest truncation error to be tolerated. Is
                               overruled by D if necessary.
        continuous_sweep     : Whether all sites should be optimised in a
                               continuous sweep from left to right or only
                               every second bond. Helpful for time evolutions.
        even_bonds           : If continuous_sweep=False, decides whether the
                               even bonds should be optimised or the odd bonds.
                               No effect for continuous_sweep=True.

        Returns the summed truncation error from all sites which may be larger
        than max_truncation_error by a factor of up to the chain length.
        '''

        # perform consistency check
        self._check_for_open_boundary_conditions()

        # Make the MPS right-canonical beforehand.
        # Without it, the resulting state is useless.
        self.make_MPS_right_canonical()

        summed_truncation_error = 0.

        if continuous_sweep:

            # compress every bond

            for l in range(self._L-1):
                D_here,trErr=self.set_twosite_tensor(l,
                                                    self.get_twosite_tensor(l),
                                                    'l',D,max_truncation_error)
                summed_truncation_error += trErr

        else:

            # Only compress every second bond.
            # But we also have to sweep through the other bond.

            mod = 0 if even_bonds else 1
            LBD = self.get_local_bond_dimensions()[1:-1] # without triv. edges

            for l in range(self._L-1):

                D_here = D if l % 2 == mod else LBD[l]
                tr_err_here = max_truncation_error if l % 2 == mod else None

                _,trEerr=self.set_twosite_tensor(l,self.get_twosite_tensor(l),
                                                 'l',D_here,tr_err_here)

                summed_truncation_error += trErr

        # return the summed truncation error
        return summed_truncation_error

    def decompose_into_state_vector(self,force=False):
        '''
        Decomposes the MPS into a dense ED state vector. If you try to
        decompose an MPS of more than 20 sites, the function will prevent this
        and will instead throw an exception. This is done to prevent filling up
        your entire memory. Instead of taking a lot of time to fill your memory
        and then throwing an out of memory error, this function skips the
        former part and throws the exception in the beginning. This is intended
        for the case that you accidentally call this function for a large MPS.
        This can be circumvented by setting force to True.
        '''

        L = self.get_length()

        # prevent usage at L>20 unless forced
        if L > 20 and not force:
            raise TypeError("The MPS is too large to be decomposed into a "
                            "dense state vector. If you still want to perform "
                            "that action set force=True.")

        # index order: s_1 | a_0 | a_1 @ s_2 | a_1 | a_2
        #           => s_1 | a_0 | s_2 | a_2
        state_vector = tensordot(self.get_site(0),self.get_site(1),
                                 axes=([2],[1]))

        for i in range(2,L):

            # index order: s_1 | a_0 | s_2 | ... | s_p-1 | a_p-1
            #            @ s_p | a_p-1 | a_p
            #           => s_1 | a_0 | s_2 | ... | s_p-1 | s_p | a_p
            state_vector = tensordot(state_vector,self.get_site(i),
                                     axes=([i+1],[1]))

        if self.get_boundary_condition() != 'OBC':

            # index order: s_1 | a_0 | s_2 | ... | s_L | a_L
            #           => s_1 | s_2 | ... | s_L
            state_vector = np.trace(state_vector,axis1=1,axis2=-1)

        # flatten result into vector and return
        if type(state_vector) is np.ndarray:
            return state_vector.flatten()
        else:
            return state_vector.reconstruct_full_tensor().flatten()

    def get_Schmidt_spectrum(self,pos=None):
        '''
        Calculates the Schmidt spectrum of the MPS for a given bipartition of
        the MPS. The bond which is cut is the bond right of the tensor at
        position pos. if pos is set to None, the bond in the middle will be
        cut. For uneven chain lengths the left bond of the two bonds in the
        middle will be cut.

        This function modifies the MPS to mixed canonical form with the border
        between left- and right-normalized site tensors being at the supplied
        position 'pos'.

        Returns a list of the Schmidt values sorted in descending order.
        '''

        # perform consistency checks
        if self._boundary != 'OBC':
            raise ValueError('This operation is not defined for non-open '
                             'boundary conditions.')

        # adapt variable pos
        if pos is None:
            pos = self.get_length()//2
        else:
            pos = pos+1
        pos = int(pos)

        # from here on, pos=0 refers to trivial bond left of the first bond

        # get MPS in mixed canonical form
        A = self.get_left_normalized_border()
        B = self.get_right_normalized_border()
        L = self.get_length()

        # move left normalized border to right if need be
        if A < pos:
            self.move_left_normalized_boundary_to_the_right(pos-A)

        # move right normalized border to left if need be
        if B > pos:
            self.move_right_normalized_boundary_to_the_left(B-pos)

        # index order: s_p | s_p+1 | a_p-1 | a_p+1 
        tensor = self.get_twosite_tensor(pos-1)

        # index order: s_p | a_p-1 | s_p+1 | a_p+1 
        tensor = tensor.swapaxes(1,2)

        # get the singular values
        # tensor index order:  s_p | a_p-1  |  s_p+1 | a_p+1
        #                  => (s_p | a_p-1) | (s_p+1 | a_p+1)
        if self._useReducedTensors:

            rt.combine_axes_for_reduced_tensor(tensor,+1,2,2)
            rt.combine_axes_for_reduced_tensor(tensor,-1,0,2)
            U,S,V,truncation_error = rt.SVD_for_reduced_matrix(tensor, D=None,
                                                     max_truncation_error=None,
                                                               normalize='b')

        else:

            shape = np.shape(tensor)
            tensor = tensor.reshape([shape[0]*shape[1],shape[2]*shape[3]])
            S = np.linalg.svd(tensor, full_matrices=False, compute_uv=False)

        # return singular values (and remove possible nonsense values)
        keep = self.get_site(pos-1).shape[2]
        return S[:keep]

    def get_entanglement_entropy(self,pos=None):
        '''
        Calculates the entanglement entropy of the MPS for a given bipartition.
        This function is essentially a wrapper for get_Schmidt_spectrum.

        Returns the entanglement entropy.
        '''

        # perform consistency checks
        if self._boundary != 'OBC':
            raise ValueError('This operation is not defined for non-open '
                             'boundary conditions.')

        S = self.get_Schmidt_spectrum(pos)
        return -np.dot(S**2,np.log2(S**2, out=np.zeros_like(S), where=(S!=0)))


    def get_entanglement_spectrum(self,pos=None):
        '''
        Calculates the entanglement spectrum of the MPS for a given
        bi-partition. This function is essentially a wrapper for
        self.get_Schmidt_spectrum().

        Returns a list of the entanglement values sorted in ascending order.
        '''

        # perform consistency checks
        if self._boundary != 'OBC':
            raise ValueError('This operation is not defined for non-open '
                             'boundary conditions.')

        S = self.get_Schmidt_spectrum(pos)
        return -np.log(S)*2


    def get_Schmidt_spectrum_all_bonds(self):
        '''
        Calculates the Schmidt values for all bonds. This function is
        essentially a wrapper for get_Schmidt_spectrum.

        Returns a list containing the Schmidt values for each site which, in
        turn, are sorted in lists.
        '''

        # perform consistency checks
        if self._boundary != 'OBC':
            raise ValueError('This operation is not defined for non-open '
                             'boundary conditions.')

        S = []
        for pos in range(self.get_length()-1):
            S.append(self.get_Schmidt_spectrum(pos))

        return S


    def get_entanglement_entropy_all_bonds(self,pos=None):
        '''
        Calculates the entanglement entropy of the MPS for a given bipartition.
        This function is essentially a wrapper for get_Schmidt_spectrum.

        Returns the entanglement entropy.
        '''

        if self._boundary != 'OBC':
            raise ValueError('This operation is not defined for non-open '
                             'boundary conditions.')

        EE = []
        for pos in range(self.get_length()-1):
            S = self.get_Schmidt_spectrum(pos)
            EE.append(-np.dot(S**2,np.log2(S**2, out=np.zeros_like(S),
                                           where=(S!=0))))

        return EE


    def get_entanglement_spectrum_all_bonds(self,pos=None):
        '''
        Calculates the entanglement spectrum of the MPS for a given
        bipartition. This function is essentially a wrapper for
        get_Schmidt_spectrum.

        Returns a list of the entanglement values sorted in ascending order.
        '''

        if self._boundary != 'OBC':
            raise ValueError('This operation is not defined for non-open '
                             'boundary conditions.')

        S = []
        for pos in range(self.get_length()-1):
            S.append(-np.log(self.get_Schmidt_spectrum(pos))*2)

        return S


    def calc_correlator_for_MPS_in_Pauli_matrix(self,op1_dict,op2_dict):
        '''
        Calculate the correlation function given by op1_dict and op2_dict for
        the MPS. The correlation function is given as <o1 o2> - <o1><o2> where
        o1 is represented by op1_dict and o2 by op2_dict. These variables are
        dictionaries where the keys represent the position of a local operator
        while the value is either 'x', 'y', 'z' or '1' representing the Pauli
        matrices, respectively the identity matrix, although setting the latter
        is unnecessary. The positions in op1_dict and op2_dict may not overlap.

        Example: To calculate the correlation in z-direction between the first
        and the last site in the MPS, set op1_dict = {0:'z'} and
        op2_dict = {-1:'z'}.

        Does not work for reduced tensors.
        '''

        sigma = {'x': np.array([[0, 1],[1, 0.]]),
                 'y': np.array([[ 0.+0.j, -0.-1.j],[ 0.+1.j,  0.+0.j]]),
                 'z': np.array([[ 1,  0],[ 0, -1.]]),
                 '1': np.array([[1, 0],[0, 1.]])}

        L = self.get_length()

        # initialise operators
        op1  = L*[None]
        op2  = L*[None]
        op12 = L*[None]

        # fill operators
        for key,value in op1_dict.items():
            op1[key]  = sigma[value]
            op12[key] = sigma[value]

        for key,value in op2_dict.items():
            op2[key] = sigma[value]
            if op12[key] is not None:
                raise ValueError('Operators must not overlap in space.')
            op12[key] = sigma[value]

        # calculate expectation values
        eval_1  = self.expectation_value(op1)
        eval_2  = self.expectation_value(op2)
        eval_12 = self.expectation_value(op12)

        return eval_12 - eval_1 * eval_2


    def calc_correlator_G_function(self,opL_list,opR_list,distance):
        '''
        Calculate the G function. The G function is the mean of all expectation
        values <o1 o2> where o1 and o2 are a block of several local operators.
        While the positions on which o1 and o2 act changes throughout the
        calculation, they always stay a distance'distance' from each other. For
        this reason, the operators are supplied in a list and a distance must
        be given. The elements of the lists opL_list and opR_list are strings
        of the form 'x', 'y', 'z' or '1' and describe the Pauli operator,
        respectively the identity matrix to be applied at this relative
        position. This function is especially useful to determine spin orders
        in certain spin directions in the presence of disorder as it provides a
        kind of disorder averaging.

        Example: opL_list = ['x'] opR_list['x','z'], distance = 5, L = 10
        Calculates: (<x11111xz11> + <1x11111xz1> + <11x11111xz>) / 3

        Does not work for reduced tensors.
        '''

        sigma = {'x': np.array([[0, 1],[1, 0.]]),
                 'y': np.array([[0.+0.j, -0.-1.j],[0.+1.j, 0.+0.j]]),
                 'z': np.array([[1, 0],[0, -1.]]),
                 '1': np.array([[1, 0],[0, 1.]])}

        L = self.get_length()
        G = 0

        # calculate all possible expectation values
        for i in range(L-distance):

            op = L*[None]

            for j,op2 in enumerate(opL_list):
                op[i+j] = sigma[op2]

            for j,op2 in enumerate(opR_list):
                op[i+j+distance] = sigma[op2]

            G += np.abs(self.expectation_value(op))

        # divide by the number of expectation values and return
        return G/(L-distance)

    def calc_total_correlator_local(self,operator,*,print_progress=False):
        '''
        Calculate for the local operator 'operator' all possible values of
        <o_pos1 o_pos2> where pos1 and pos2 are all possible positions in the
        spin chain as long as pos1 < pos2. This is a relatively demanding
        numerical task which can be greatly speed up by using the
        infrastructure of the L-tensors and R-tensors as done here. Since this
        may be a lengthy operation, the key-word argument 'print_progress' may
        be set to True to print the current state of the calculation to the
        standard output. The given operator must have the index
        order: s_p | s'_p.

        Returns a matrix of shape (L,L) with L being the length of the spin
        chain. This matrix contains all expectation values in the upper
        triangular half. The lower triangular half and the main diagonal
        contain np.nan instead.
        '''

        L = self.get_length()
        cor = np.empty([L,L])*np.nan

        #calculate all R tensors up to site 2 (counting form 0)

        # index order: a'_0, a_0
        Rtensor = [np.array([[1.0]])]
        for i in range(L-2):
            site = -i-1

            # index order: a'_p | a_p @ s_p | a_p-1 | a_p => a'_p | s_p | a_p-1
            R = np.tensordot(Rtensor[-1],self.get_site(site),axes=([1],[2]))

            # index order: s_p | a'_p | a'_p-1 @ a'_p | s_p | a_p-1
            #           => a'_p-1 | a_p-1
            R = np.tensordot(np.conjugate(self.get_site(site)).swapaxes(-1,-2),
                             R,axes=([0,1],[1,0]))

            Rtensor.append(R)

        counter = 0
        counter_max = L*(L-1)//2

        # index order: a'_0 | a_0
        L1 = np.array([[1.0]])

        # first loop through chain
        for l in range(0,L-1):

            # index order: s_l | s'_l @ s_l | a_l-1 | a_l => s'_l | a_l-1 | a_l
            L2 = np.tensordot(operator,self.get_site(l),axes=((0,),(0,)))

            # index order: a'_l-1 | a_l-1 @ s'_l | a_l-1 | a_l
            #           => a'_l-1 | s'_l | a_l
            L2 = np.tensordot(L1,L2,axes=((1,),(1,)))

            # index order: s'_l | a'_l | a'_l-1 @ a'_l-1 | s'_l | a_l
            #           => a'_l | a_l
            L2 = np.tensordot(np.conjugate(self.get_site(l)).swapaxes(-1,-2),
                              L2,axes=([0,2],[1,0]))

            # second loop through chain
            for r in range(l+1,L):

                counter += 1

                if print_progress:
                    print('\r[{}] [{}/{}] {}%'.format(
                        ((counter*50)//counter_max*'#').ljust(50),counter,
                         counter_max,counter*100//counter_max),end='')

                #calculate expectation value

                # index order: s_r | s'_r @ s_r | a_r-1 | a_r
                #           => s'_r | a_r-1 | a_r
                O = np.tensordot(operator,self.get_site(r),axes=((0),(0)))

                # index order: a'_r-1 | a_r-1 @ s'_r | a_r-1 | a_r
                #           => a'_r-1 | s'_r | a_r
                O = np.tensordot(L2,O,axes=((1),(1)))

                # index order: a'_r-1 | s'_r | a_r @ s'_r | a'_r | a'_r-1
                #           => a_r | a'_r
                O = np.tensordot(O,
                                np.conjugate(self.get_site(r)).swapaxes(-1,-2),
                                 axes=((1,0),(0,2)))

                # index order: a_r | a'_r @ a'_r | a_r => scalar
                O = np.tensordot(O,Rtensor[L-r-1],axes=((0,1),(1,0)))

                cor[l,r] = O

                if r == L-1:
                    continue

                # index order: a'_r-1 | a_r-1 @ s_r | a_r-1 | a_r
                #           => a'_r-1 | s_r | a_r
                L2 = np.tensordot(L2,self.get_site(r),axes=((1),(1)))

                # index order: s'_r | a'_r | a'_r-1 @ a'_r-1 | s_r | a_r
                #           => a'_r | a_r
                L2=np.tensordot(np.conjugate(self.get_site(r)).swapaxes(-1,-2),
                                L2,axes=([0,2],[1,0]))

            if l == L-2:
                continue

            #prepare L1 for next run
            del Rtensor[-1]

            # index order: a'_l-1 | a_l-1 @ s_l | a_l-1 | a_l
            #           => a'_l-1 | s_l | a_l
            L1 = np.tensordot(L1,self.get_site(l),axes=((1),(1)))

            # index order: s_l | a'_l | a'_l-1 @ a'_l-1 | s_l | a_l
            #           => a'_l | a_l
            L1 = np.tensordot(np.conjugate(self.get_site(l)).swapaxes(-1,-2),
                              L1,axes=([0,2],[1,0]))

        # print newline so that prompt does not appear midline
        if print_progress:
            print()

        # return correlation function
        return cor


    def enlarge(self,D):
        '''
        Enlarges the MPS by padding it with zeros to the new local bond
        dimension. If the new local bond dimension is smaller than the current
        one, the MPS will not be compressed but individual bonds which are
        smaller will be increased. If the MPS has open boundary conditions, the
        exponential decay toward the edges is ensured. Cannot be used for
        reduced tensors.
        '''

        if type(self.get_site(0)) is rt.reducedTensor:
            raise TypeError('This function cannot be used with reducedTensor.')

        self.set_local_bond_dimension(D)

        L  = self.get_length()
        d  = self.get_local_state_space_dimension()
        bc = self.get_boundary_condition()

        edge = int(np.log2(D))

        for pos in range(L):

            tensor = self.get_site(pos)

            if bc == 'PBC':
                D_left,D_right = D,D
            if bc == 'OBC' and pos < L//2:
                D_left  = D if pos-1 >= edge else d**pos
                D_right = D if pos   >= edge else d**(pos+1)
            elif bc == 'OBC' and pos >= L//2:
                D_left  = D if L-pos-1 >= edge else d**(L-pos)
                D_right = D if L-pos-2 >= edge else d**(L-pos-1)

            pad_axis1 = max(0,D_left  - tensor.shape[1])
            pad_axis2 = max(0,D_right - tensor.shape[2])
            pad_tuple = ((0,0),(0,pad_axis1),(0,pad_axis2))

            tensor = np.pad(tensor, pad_tuple, 'constant', constant_values=0)

            self._MPS[pos] = tensor

    def get_transfere_operator(self,pos):
        '''
        Calculates and returns the transfere operator for a given site.
        The transfere operator has shape (DxDxDxD) with D being the local bond
        dimension and the index ordering is: a_p | a'_p | a_p+1 | a'_p+1.
        '''

        # index order: s_p | a_p-1 | a_p
        M = self.get_site(pos)

        # index order: s_p | a_p-1 | a_p @ s_p | a'_p-1 | a'_p
        #           => a_p-1 | a_p | a'_p-1 | a'_p
        E = tensordot(M,np.conj(M),axes=((0),(0)))

        # index order: a_p-1 | a'_p-1 | a_p | a'_p
        E = E.swapaxes(1,2)

        return E

    ###################################################################
    # internal functions not meant to be used from outside this class #
    ###################################################################

    def _dagger(self,tensor):
        '''
        Conjugates a given matrix or tensor and swaps the last two dimensions.
        '''

        self._check_if_array(tensor)
        self._check_if_tensor_has_right_number_of_dimensions(tensor,2,'>=')

        return np.conjugate(tensor.swapaxes(-1,-2))

    def _check_if_position_is_valid(self,pos):
        '''
        Checks whether a given position exists in the MPS. If yes, nothing
        happens, if not, a ValueError is raised.
        '''

        if pos > self.get_length() - 1 or pos < -self.get_length():
            raise ValueError('Element {} cannot be accessed, MPS '.format(pos)+
                             'only consists of {}'.format(self.get_length())+
                             ' sites.')

    def _check_if_array(self,tensor):
        '''
        Checks whether the given object 'tensor' is of type np.ndarray or, if
        used, of type rt.reducedTensor. If it is, nothing happens, if not, a
        TypeError is raised.
        '''

        if type(tensor) is np.ndarray:
            return
        elif self._useReducedTensors and type(tensor) is rt.reducedTensor:
            return

        raise TypeError("The given object is not a numpy.ndarray or a "
                        "rt.reducedTensor but a {}.".format(type(tensor)))

    def _check_if_tensor_has_right_number_of_dimensions(self,tensor,
                                                        number_of_dimensions,
                                                        compare_string='='):
        '''
        Checks whether the given tensor has the right number of dimensions. If
        yes, nothing happens, if not, a ValueError is raised.
        '''

        nod = number_of_dimensions
        dim = np.ndim(tensor)
        
        if compare_string == '=' and dim != nod:
            raise TypeError("The given tensor must be a tensor of rank "
                            "{} but it has rank {}.".format(nod,dim))
        elif compare_string == '!=' and dim == nod:
            raise TypeError("The given tensor must not be a tensor of rank "
                            "{} but it has rank {}.".format(nod,dim))
        elif compare_string == '>' and dim <= nod:
            raise TypeError("The given tensor must be a tensor of rank "
                            ">{} but it has rank {}.".format(nod,dim))
        elif compare_string == '<' and dim >= nod:
            raise TypeError("The given tensor must be a tensor of rank "
                            "<{} but it has rank {}.".format(nod,dim))
        elif compare_string == '<=' and dim > nod:
            raise TypeError("The given tensor must be a tensor of rank "
                            "<={} but it has rank {}.".format(nod,dim))
        elif compare_string == '>=' and dim < nod:
            raise TypeError("The given tensor must be a tensor of rank "
                            ">={} but it has rank {}.".format(nod,dim))

    def _check_for_open_boundary_conditions(self):
        '''
        Checks whether the chosen boundary conditions are open. If not, an
        error is raised. Periodic boundary conditions will lead to a lot of
        operations and concepts to be ill-defined.
        '''

        if self._boundary != 'OBC':
            raise ValueError('This operation is not defined for non-open '
                             'boundary conditions.')

    def _check_if_external_MPS_is_compatible(self,MPS):
        '''
        Checks whether it is possible that both MPSs stem from the same
        Hilber-space. For this to be true, the length L and the local space
        state dimension d must be the same. If both MPS are not compatible, a
        ValueError will be thrown.
        '''

        d1 = self.get_local_state_space_dimension()
        d2 =  MPS.get_local_state_space_dimension()

        if d1 != d2:
            raise ValueError('Parameter d of both MPS must be equal '
                             '({} != {}).'.format(d1,d2))

        L1 = self.get_length()
        L2 =  MPS.get_length()

        if L1 != L2:
            raise ValueError('Parameter L of both MPS must be equal '
                             '({} != {}).'.format(L1,L2))

    def _check_if_external_operator_string_is_compatible(self,operator_string):
        '''
        Checks whether a given operator string is compatible to the MPS. For
        this, the length L must be equal and each operator in the string must
        be of shape (d,d).
        '''

        if len(operator_string) != self.get_length():
            raise ValueError('The operator string must be of length '
                           '{} (length of the MPS) '.format(self.get_length())+
                           'but is of length {}'.format(len(operator_string)))

        for i in range(len(operator_string)):

            if operator_string[i] is None:
                continue

            self._check_if_array(operator_string[i])
            self._check_if_tensor_has_right_number_of_dimensions(
                                                          operator_string[i],2)
            d = self.get_local_state_space_dimension()

            if np.shape(operator_string[i]) != (d,d):
                dim1 = np.shape(operator_string[i])[0]
                dim2 = np.shape(operator_string[i])[1]
                raise ValueError('Local operators must be of shape '
                               '({},{}) but operator at position '.format(d,d)+
                               '{} is of shape ({},{})'.format(i,dim1,dim2))

    def save_hdf5(self,hdf5_handler,name):
        '''
        Saves a given MPS in a HDF5 file to a given position.
        hdf5_handler: The handler to the HDF5 file (from h5py). It can point
                      to a group within the file.
        MPS_:         The MPS object to be saved.
        name:         A new group under this name will be created. In this
                      group, the MPS will be saved. This group has to be
                      referenced in the loading function to load the MPS again.
        '''

        # create group
        hdf5_handler.create_group(name)
        f = hdf5_handler[name]

        # save attributes of the MPS
        f.attrs['_d'] = self._d
        f.attrs['_L'] = self._L
        f.attrs['_D'] = self._D

        f.attrs['_multiplicative_factor'] = self._multiplicative_factor
        f.attrs['_normalisation_tolerance'] = self._normalisation_tolerance
        f.attrs['initial_truncation_error'] = self.initial_truncation_error
        f.attrs['_useReducedTensors'] = self._useReducedTensors

        f.attrs['_A_border'] = self._A_border
        f.attrs['_B_border'] = self._B_border

        f.attrs['_boundary'] = self._boundary

        # save site tensors of the MPS
        if self._useReducedTensors:
            for i in range(self.get_length()):
                rt.save_reducedTensor_hdf5(f,self.get_site(i),
                                           'site{}'.format(i))
        else:
            for i in range(self.get_length()):
                f['site{}'.format(i)] = self.get_site(i)
