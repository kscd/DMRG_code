import numpy as np
import scipy as sp
import copy
import scipy.sparse.linalg as linalg
from MPS import tensordot
from MPS import perform_np_SVD
from MPS import _give_convenient_numpy_float_complex_type
from DMRG_LO import *

# import reducedTensor class if it exists
# if not, only dense tensors can be used
try:
    import reducedTensor as rt
    _reducedTensorExists = True
except ImportError:
    _reducedTensorExists = False


class DMRG:
    '''
    The class 'DMRG' is a class performing the DMRG calculation and holding all
    necessary data for it. In this sense, it is an abstract class because it
    does not represent something one can 'touch'. This class has the ability to
    perform the DMRG but also a number of different algorithms akin to it. This
    includes the ability to perform DMRG-X optimisation steps to calculate
    excited states but also to allow the target state to be held orthogonal to
    a number of provided MPSs. This is helpful to calculate second ground
    states or first excited states. Furthermore, it is possible to optimise
    either one site or two sites in each optimisation step. Even more, for the
    DMRG optimisation step, it is possible to make use of a more efficient
    contraction scheme which makes the DMRG run in the third power of the local
    bond dimension instead of the fourth. This class provides full support for
    MPSs where the site tensors are either of type np.ndarray or of type
    rt.reducedTensor. This class is only able to work on instances of
    MPS_system where open boundary conditions are used.
    '''

    def __init__(self,MPS_system,method='g',sweep_tolerance=0,
                 orthogonal_MPSs=[]):
        '''
        Initialises the DMRG class. This is done by providing an instance of
        the class MPS_system on which the DMRG will be performed and a number
        of other arguments. The other arguments available are as follows:
        method          : Variable not used. Default is 'g'.
        sweep_tolerance : Default is 0.
        orthogonal_MPSs : A list containing all the MPSs to which this MPS
                          should be held orthogonal to. Default is [] meaning
                          that this MPS should not be held orthogonal to other
                          MPSs.
        '''

        # set attributes
        self.system = MPS_system
        self.system.set_LR_driver('manual')
        self.sweep_tolerance = sweep_tolerance
        self.stay_double = False

        # for orthogonal MPSs
        self._use_orthogonal_MPSs = False if orthogonal_MPSs == [] else True
        self.orthogonal_MPSs = orthogonal_MPSs
        self._L_border_orthogonal = 0
        self._R_border_orthogonal = -1
        self._L_tensors_orthogonal = [[np.array([[1.0]])]
                                      for _ in orthogonal_MPSs]
        self._R_tensors_orthogonal = [[np.array([[1.0]])]
                                      for _ in orthogonal_MPSs]

    def activate_orthogonal_MPSs(self):
        '''
        Activates the usage of the orthogonal MPSs. This means that in a future
        run of the function 'finite_sweep', the target state will be held
        orthogonal to the MPSs provided during the initialisation.
        '''

        self._use_orthogonal_MPSs = True

    def deactivate_orthogonal_MPSs(self):
        '''
        Deactivates the usage of the orthogonal MPSs. This means that in a
        future run of the function 'finite_sweep', the target state will not be
        held orthogonal to the MPSs provided during the initialisation.
        '''

        self._use_orthogonal_MPSs = False

    def get_use_orthogonal_MPSs(self):
        '''
        Returns whether orthogonal MPSs will be used in a future run of the
        function 'finite_sweep'.
        '''

        return self._use_orthogonal_MPSs

    def _set_L_tensor_border_orthogonal(self,pos):
        '''
        Set the border of the L tensors of the MPS-MPS environments used when
        holding the target state orthogonal to the provided MPSs to be at
        position 'pos'. Negative indices are allowed to count from the right
        end of the spin chain. Because of this 'pos' may lie in the interval
        [-L-1,L] if L is the length of the spin chain. If 'pos' is positive,
        the last L tensor incorporates the site tensor at position 'pos'.
        '''

        # determine old and new L tensor border using positive indices
        old_pos=(self._L_border_orthogonal + self.system.get_length() + 1 if
                 self._L_border_orthogonal < 0 else self._L_border_orthogonal)

        new_pos = pos + self.system.get_length() + 1 if pos < 0 else pos

        self._L_border_orthogonal = pos

        # delete L tensors as needed
        if old_pos == new_pos:
            # nothing changes
            return
        if old_pos > new_pos:
            # no new tensors, delete old tensors
            delete = old_pos - new_pos
            for i in range(len(self.orthogonal_MPSs)):
                del self._L_tensors_orthogonal[i][-delete:]
            return


        # calculate new L tensors as needed
        num = new_pos - old_pos

        # loop over orthogonal MPSs
        for i,MPS in enumerate(self.orthogonal_MPSs):

            # loop over new L tensors
            for _ in range(num):

                site = len(self._L_tensors_orthogonal[i]) - 1

                if site >= self.system.get_length():
                    raise ValueError('Cannot calculate next L-tensor. All '
                                     'L-tensors are already calculated.')
                elif site == -1:

                    # set last L tensor as dummy tensor
                    if self._useReducedTensors:
                        q_L0 = self.system._MPS[0].q_vectors[1][0]
                        q_L1 = MPS._MPS[0].q_vectors[0][0]
                        self._L_tensors_orthogonal[i] = [
                            rt.reducedTensor({(q_L0,q_L1): np.array([[1.0]])},
                                             [(q_L0,),(q_L1,)],[-1,1],
                                             sectors_given=True)]
                    else:
                        self._L_tensors_orthogonal[i] = [np.array([[1.0]])]
                    continue

                # perform contraction

                # index order: c_p-1 | a_p-1 @ s_p | a_p-1 | a_p
                #           => c_p-1 | s_p | a_p
                L = tensordot(self._L_tensors_orthogonal[i][-1],
                              self.system.get_site(site),axes=([1],[1]))

                # index order: s_p | c_p | c_p-1 @ c_p-1 | s_p | a_p
                #           => c_p | a_p
                L = tensordot(self.system._dagger(
                  self.orthogonal_MPSs[i].get_site(site)),L,axes=([0,2],[1,0]))

                # set new L-tensor
                self._L_tensors_orthogonal[i].append(L)

    def _set_R_tensor_border_orthogonal(self,pos):
        '''
        Set the border of the R tensors of the MPS-MPS environments used when
        holding the target state orthogonal to the provided MPSs to be at
        position 'pos'. Negative indices are allowed to count from the right
        end of the spin chain. Because of this 'pos' may lie in the interval
        [-L-1,L] if L is the length of the spin chain. If 'pos' is positive,
        the last L tensor incorporates the site tensor at position 'pos+1'.
        '''

        # determine old and new L tensor border using positive indices
        old_pos=(self._R_border_orthogonal + self.system.get_length() + 1 if
                 self._R_border_orthogonal < 0 else self._R_border_orthogonal)

        new_pos = pos + self.system.get_length() + 1 if pos < 0 else pos

        self._R_border_orthogonal = new_pos

        # delete R tensors as needed
        if old_pos == new_pos:
            # nothing changes
            return
        elif old_pos < new_pos:
            # no new tensors, delete old tensors
            delete = new_pos - old_pos
            for i in range(len(self.orthogonal_MPSs)):
                del self._R_tensors_orthogonal[i][-delete:]


        # calculate new R tensors as needed
        num = old_pos - new_pos

        # loop over orthogonal MPSs
        for i,MPS in enumerate(self.orthogonal_MPSs):

            # loop over new R tensors
            for _ in range(num):

                site = (self.system.get_length() -
                        len(self._R_tensors_orthogonal[i]))

                if site < 0:
                    raise ValueError('Cannot calculate next R-tensor. All '
                                     'R-tensors are already calculated.')
                elif site == self.system.get_length():

                    # set last R tensor as dummy tensor
                    if self._useReducedTensors:
                        q_R0 = self._MPS[-1].q_vectors[2][0]
                        q_R1 = self.MPS._MPS[0].q_vectors[1][0]
                        self._R_tensors_orthogonal[i] = [
                            rt.reducedTensor({(q_R0,q_R1): np.array([[1.0]])},
                                             [(q_R0,),(q_R1,)],[1,-1,],
                                             sectors_given=True)]
                    else:
                        self._R_tensors_orthogonal[i] = [np.array([[1.0]])]
                    continue

                # perform contraction

                # index order: c_p | a_p @ s_p | a_p-1 | a_p
                #           => c_p | s_p | a_p-1
                R = tensordot(self._R_tensors_orthogonal[i][-1],
                              self.system.get_site(site),axes=([1],[2]))

                # index order: s_p | c_p | c_p-1 @ c_p | s_p | a_p-1
                #           => c_p-1 | a_p-1
                R = tensordot(self.system._dagger(
                  self.orthogonal_MPSs[i].get_site(site)),R,axes=([0,1],[1,0]))

                # set new R-tensor
                self._R_tensors_orthogonal[i].append(R)


    def enlarge(self,D):
        '''
        Enlarges the MPS stored in the provided instance of MPS_system by
        padding it with zeros to the new local bond dimension. In addition,
        pad all the L and R tensors used for holding this MPS orthogonal to the
        MPSs provided earlier accordingly. If the new local bond dimension is
        smaller than the current one, the MPS will not be compressed but
        individual bonds which are smaller will be increased. If the MPS has
        open boundary conditions, the exponential decay toward the edges is
        ensured. Cannot be used for reduced tensors. If this MPS should be held
        orthogonal to no other MPSs, this function just calls the function
        'enlarge' of the stored instance of MPS_system and does nothing else.
        '''

        # enlarge MPS in stored MPS_system
        self.system.enlarge(D)

        # loop over orthogonal MPSs
        for i in range(len(self.orthogonal_MPSs)):

            # loop over L tensors
            for pos in range(len(self._L_tensors_orthogonal[i])):

                padD = (max(0,self.system.get_site(pos).shape[1]
                            - self._L_tensors_orthogonal[i][pos].shape[1]))

                self._L_tensors_orthogonal[i][pos] = np.pad(
                    self._L_tensors_orthogonal[i][pos], ((0,0),(0,padD)),
                    'constant', constant_values=0)

            # loop over R tensors
            for pos in range(len(self._R_tensors_orthogonal[i])):

                padD = (max(0,self.system.get_site(-pos).shape[1]
                            - self._R_tensors_orthogonal[i][pos].shape[1]))

                self._R_tensors_orthogonal[i][pos] = np.pad(
                    self._R_tensors_orthogonal[i][pos], ((0,0),(0,padD)),
                    'constant', constant_values=0)


    def finite_sweep(self,sweeps,two_site,method='g',fast=False,ortho2=False,
                     orthoDelta=10,reinitialiseOrthogonal=False,
                     *,print_steps=False):
        '''
        Performs a number of DMRG sweeps on the instance of MPS_system stored
        in the present instance of this class. To do this, the MPS will be
        brought into left- or right-canonical form if this has not already been
        done so. The precise canonical form selected, which also influences the
        sweep direction the DMRG starts with, will be selected based on the
        smaller amount of work that has to be carried out to bring the MPS into
        canonical form. It is not possible to perform only partial sweeps with
        this function, only complete sweeps are possible. It is possible to use
        reduced tensors with this function. To do this, no further action is
        required. It is only required that the site tensors of the stored MPS
        and MPO are instances of reducedTensors. A great number of adjustment
        for this function are available based on individual needs. These
        possibilities are explained below where the arguments this function
        supports are discussed.

        This function takes the following arguments:

        sweeps                 : The number of sweeps to be performed.
        two_site               : Decide whether one-site DMRG (two_site=False)
                                 or two-site DMRG (two_site=True) should be
                                 used.
        method                 : The precise method used in the optimisation
                                 step. Default is 'g'. Options are given below.
        fast                   : Whether the D^4 scaling or the D^3 scaling
                                 should be used. Default is false, indicating
                                 the D^4 method.
        ortho2                 : If this MPS should be held orthogonal to other
                                 MPSs, decides the precise mechanism to do so.
                                 Default is False. False results in raising the
                                 energy of all involved MPSs by the value of
                                 'orthoDelta', while setting it to True results
                                 of all these energies being implicitly set to
                                 zero.
        orthoDelta             : The energy by which the MPSs given
                                 orthogonal_MPS are raised. Only applicable if
                                 ortho2=False. Default is 10.
        reinitialiseOrthogonal : If the MPSs to which this MPS should be held
                                 orthogonal to have been changed since the last
                                 time the orthogonal environments have been
                                 created, the orthogonal L/R tensors have to be
                                 recreated. This can be achieved by setting
                                 reinitialiseOrthogonal=True. Default is False.
        print_steps            : Whether progress should be printed to the
                                 standard output. Default is False.

        The variable 'method' may be set to the following values:

        'g'     : Optimise toward ground state (traditional DMRG step).
        'h'     : Optimise toward highest energy eigenstate
                  (same as 'g' for -H).
        'gED'   : Same as 'g' but uses exact diagonalisation. Does not use old
                  site tensor as initial guess. Can only by used if H is of
                  type np.ndarray or rt.reducedTensor.
        'hED'   : Same as 'h' but uses exact diagonalisation. Does not use old
                  site tensor as initial guess. Can only by used if H is of
                  type np.ndarray or rt.reducedTensor.
        'DMRGX' : To perform a DMRG-X step. Is not optimised toward ground
                  state but toward eigenstate closest in terms of spatial
                  overlap to old site tensor. Can only by used if H is of type
                  np.ndarray or rt.reducedTensor.

        This function returns a np.ndarray of shape (3,sweeps). It contains for
        each performed sweep the energy, energy variance and summed truncation
        error.
        '''

        # perform consistency checks:
        if fast == True and method in ['gED','hED','DMRGX']:
            raise ValueError("The option fast=True may not be used together "
                             "with method={}.".format(method))

        # reinitialise orthogonal L/R tensors if needed
        if reinitialiseOrthogonal:
            self._L_border_orthogonal = 0
            self._R_border_orthogonal = -1
            self._L_tensors_orthogonal = [[np.array([[1.0]])]
                                          for _ in self.orthogonal_MPSs]
            self._R_tensors_orthogonal = [[np.array([[1.0]])]
                                          for _ in self.orthogonal_MPSs]

        # get properties of MPS
        length = self.system.get_length()
        A = self.system.get_left_normalized_border()
        B = self.system.get_right_normalized_border()
        sites = 2 if two_site else 1

        # result_array contains energy, energy variance
        # and summed truncation_error per sweep
        if two_site:
            result_array = np.zeros([3,sweeps])
            result_array[:2] = np.nan
        else:
            result_array = np.zeros([3,sweeps])*np.nan

        # find out initial sweep direction
        if B == 1:
            # MPS is right-canonical, sweep to the right
            direction = 'R'
        elif A == length-1:
            # MPS is left-canonical, sweep to the left
            direction = 'L'
        else:
            # MPS is not canonical, make it canonical based on least effort
            if B < length-A:
                self.system.make_MPS_right_canonical()
                direction = 'R'
            else:
                self.system.make_MPS_left_canonical()
                direction = 'L'

        # loop over number of sweeps, optimisation starts here
        for sweep in range(sweeps):

            # prepare for sweep
            if direction == 'R':
                sweep_range = range(length-sites+1)
                self.system.set_L_tensor_border(0)
                self.system.set_R_tensor_border(sites)

                if self.get_use_orthogonal_MPSs():
                    self._set_L_tensor_border_orthogonal(0)
                    self._set_R_tensor_border_orthogonal(sites)

            elif direction == 'L':
                sweep_range = range(length-sites,-1,-1)
                self.system.set_L_tensor_border(length-sites)
                self.system.set_R_tensor_border(length)

                if self.get_use_orthogonal_MPSs():
                    self._set_L_tensor_border_orthogonal(length-sites)
                    self._set_R_tensor_border_orthogonal(length)

            # loop over sites in MPS, sweep through the system
            for site in sweep_range:

                # Build effective Hamiltonian H and get initial guess M
                if two_site:

                    # index order: s_p | s_p+1 | a_p-1 | a_p+1
                    M = self.system.get_twosite_tensor(site)

                    # index order: b_p-1 | b_p+1 | s'_p | s_p | s'_p+1 | s_p+1
                    W = self.system.get_MPO_twosite_tensor(site)

                else:

                    # index order: s_p | a_p-1 | a_p
                    M = copy.deepcopy(self.system.get_site(site))

                    # index order: b_p-1 | b_p | s'_p | s_p
                    W = self.system.get_MPO_site(site)

                # index order: a_p-1 | b_p-1 | a'_p-1
                L = self.system.get_L_tensor(-1)

                # one-site index order: a_p | b_p | a'_p
                # two-site index order: a_p+1 | b_p+1 | a'_p+1
                R = self.system.get_R_tensor(-1)

                if fast:

                    # D^3 scaling with abstract effective Hamiltonian

                    t = _give_convenient_numpy_float_complex_type(L.dtype.type,
                                                                  R.dtype.type)
                    t = _give_convenient_numpy_float_complex_type(t,
                                                                  W.dtype.type)

                    # receive effective Hamiltonian

                    # 1s io: s'_p | a'_p-1 | a'_p | s_p | a_p-1 | a_p
                    # 2s io: s'_p|s'_p+1|a'_p-1|a'_p+1|s_p|s_p+1|a_p-1|a_p+1
                    if self.system._useReducedTensors:
                        H = DMRG_LO_REDUCED(L,R,W,M,dtype=np.dtype(t))
                    else:
                        H = DMRG_LO(L,R,W,dtype=np.dtype(t))

                else:

                    # D^4 scaling with direct effective Hamiltonian

                    # 1s io: b_p-1 | b_p | s'_p | s_p @ a_p | b_p | a'_p
                    #     => b_p-1 | s'_p | s_p | a_p | a'_p
                    # 2s io: b_p-1 | b_p+1 | s'_p | s_p | s'_p+1 | s_p+1
                    #      @ a_p+1 | b_p+1 | a'_p+1
                    #   => b_p-1 | s'_p | s_p | s'_p+1 | s_p+1 | a_p+1 | a'_p+1
                    H = tensordot(W,R,axes=([1],[1]))

                    # 1s io: a_p-1 | b_p-1 | a'_p-1
                    #      @ b_p-1 | s'_p | s_p | a_p | a'_p
                    #     => a_p-1 | a'_p-1 | s'_p | s_p | a_p | a'_p
                    # 2s io: a_p-1 | b_p-1 | a'_p-1
                    #    @ b_p-1 | s'_p | s_p | s'_p+1 | s_p+1 | a_p+1 | a'_p+1
                    # => a_p-1| a'_p-1| s'_p| s_p| s'_p+1| s_p+1| a_p+1| a'_p+1
                    if self.system._useReducedTensors:

                        # only keep sectors which are needed
                        H = rt.contract_reduced_tensors_along_axes(L,H,[1],[0],
                                               capture_garden_eden=False,
                                               for_DMRG = 2 if two_site else 1)

                    else:

                        H = tensordot(L,H,axes=([1],[0]))


                    if two_site:

                        if self.system._useReducedTensors:

                            #io a_p-1|a'_p-1|s'_p|s_p|s'_p+1|s_p+1|a_p+1|a'_p+1
                            #=> s'_p|a'_p-1|a_p-1|s_p|s'_p+1|s_p+1|a_p+1|a'_p+1
                            rt.swapaxes_for_reduced_tensor_inplace(H,0,2)

                            #io s'_p|a'_p-1|a_p-1|s_p|s'_p+1|s_p+1|a_p+1|a'_p+1
                            #=> s'_p|s'_p+1|a_p-1|s_p|a'_p-1|s_p+1|a_p+1|a'_p+1
                            rt.swapaxes_for_reduced_tensor_inplace(H,1,4)

                            #io s'_p|s'_p+1|a_p-1|s_p|a'_p-1|s_p+1|a_p+1|a'_p+1
                            #=> s'_p|s'_p+1|a'_p-1|s_p|a_p-1|s_p+1|a_p+1|a'_p+1
                            rt.swapaxes_for_reduced_tensor_inplace(H,2,4)

                            #io s'_p|s'_p+1|a'_p-1|s_p|a_p-1|s_p+1|a_p+1|a'_p+1
                            #=> s'_p|s'_p+1|a'_p-1|a'_p+1|a_p-1|s_p+1|a_p+1|s_p
                            rt.swapaxes_for_reduced_tensor_inplace(H,3,7)

                            #io s'_p|s'_p+1|a'_p-1|a'_p+1|a_p-1|s_p+1|a_p+1|s_p
                            #=> s'_p|s'_p+1|a'_p-1|a'_p+1|s_p|s_p+1|a_p+1|a_p-1
                            rt.swapaxes_for_reduced_tensor_inplace(H,4,7)

                            #io s'_p|s'_p+1|a'_p-1|a'_p+1|s_p|s_p+1|a_p+1|a_p-1
                            #=> s'_p|s'_p+1|a'_p-1|a'_p+1|s_p|s_p+1|a_p-1|a_p+1
                            rt.swapaxes_for_reduced_tensor_inplace(H,6,7)

                        else:

                            # index order same as above
                            H = H.swapaxes(0,2)
                            H = H.swapaxes(1,4)
                            H = H.swapaxes(2,4)
                            H = H.swapaxes(3,7)
                            H = H.swapaxes(4,7)
                            H = H.swapaxes(6,7)

                        dimH = np.shape(H)

                        #io: s'_p|s'_p+1|a'_p-1|a'_p+1 | s_p|s_p+1|a_p-1|a_p+1
                        #=> (s'_p|s'_p+1|a'_p-1|a'_p+1)|(s_p|s_p+1|a_p-1|a_p+1)
                        if self.system._useReducedTensors:
                            rt.combine_axes_for_reduced_tensor(H,-1,4,4)
                            rt.combine_axes_for_reduced_tensor(H,+1,0,4)
                        else:
                            H = H.reshape(dimH[0]*dimH[1]*dimH[2]*dimH[3],
                                          dimH[4]*dimH[5]*dimH[6]*dimH[7])

                    else:

                        if self.system._useReducedTensors:

                            # io: a_p-1 | a'_p-1 | s'_p | s_p | a_p | a'_p
                            #  => s'_p | a'_p-1 | a_p-1 | s_p | a_p | a'_p
                            rt.swapaxes_for_reduced_tensor_inplace(H,0,2)

                            # io: s'_p | a'_p-1 | a_p-1 | s_p | a_p | a'_p
                            #  => s'_p | a'_p-1 | a'_p | s_p | a_p | a_p-1
                            rt.swapaxes_for_reduced_tensor_inplace(H,2,5)

                            # io: s'_p | a'_p-1 | a'_p | s_p | a_p | a_p-1
                            #  => s'_p | a'_p-1 | a'_p | s_p | a_p-1 | a_p
                            rt.swapaxes_for_reduced_tensor_inplace(H,4,5)

                        else:

                            # index order same as above
                            H = H.swapaxes(0,2)
                            H = H.swapaxes(2,5)
                            H = H.swapaxes(4,5)

                        dimH = np.shape(H)

                        # io:  s'_p | a'_p-1 | a'_p  |  s_p | a_p-1 | a_p
                        #  => (s'_p | a'_p-1 | a'_p) | (s_p | a_p-1 | a_p)
                        if self.system._useReducedTensors:
                            rt.combine_axes_for_reduced_tensor(H,-1,3,3)
                            rt.combine_axes_for_reduced_tensor(H,+1,0,3)
                        else:
                            H = H.reshape(dimH[0]*dimH[1]*dimH[2],
                                          dimH[3]*dimH[4]*dimH[5])


                # modify eff. Hamiltonian by considering orth. MPSs if needed
                if (self.get_use_orthogonal_MPSs() and
                    len(self.orthogonal_MPSs) > 0 and not ortho2):

                    # set energies of orthogonal MPSs to zero
                    H = self.modify_effective_Hamiltonian(H,site,two_site,fast)

                if (self.get_use_orthogonal_MPSs()
                    and len(self.orthogonal_MPSs) > 0 and ortho2):

                    # elevate energies of orthogonal MPSs by 'orthoDelta'
                    H = self.modify_effective_Hamiltonian2(H,site,two_site,
                                                           fast,
                                                           Delta=orthoDelta)


                # perform optimisation
                # M one-site index order: s_p | a_p-1 | a_p
                # M two-site index order: s_p | s_p+1 | a_p-1 | a_p+1
                w,M = self._optimisation_step(H,M,method=method)
                del H

                # change site tensor in MPS to new site tensor
                # and update corresponding information
                if two_site:

                    if direction == 'L':

                        if site > 0:

                            self.system.set_L_tensor_border(site-1)

                            if self.get_use_orthogonal_MPSs():
                                self._set_L_tensor_border_orthogonal(site-1)

                            D, t = self.system.set_twosite_tensor(site,M,'r')

                            self.system.set_R_tensor_border(site+1)

                            if self.get_use_orthogonal_MPSs():
                                self._set_R_tensor_border_orthogonal(site+1)

                        else:

                            D, t = self.system.set_twosite_tensor(site,M,'r')

                    else:

                        if site < length - 2:

                            self.system.set_R_tensor_border(site+3)

                            if self.get_use_orthogonal_MPSs():
                                self._set_R_tensor_border_orthogonal(site+3)

                            D, t = self.system.set_twosite_tensor(site,M,'l')

                            self.system.set_L_tensor_border(site+1)

                            if self.get_use_orthogonal_MPSs():
                                self._set_L_tensor_border_orthogonal(site+1)

                        else:

                            D, t = self.system.set_twosite_tensor(site,M,'l')

                    # store truncation error in results_array
                    result_array[2,sweep] += t

                else:

                    self.system.set_site(site,M)

                    if direction == 'L':
                        if site > 0:

                            self.system.set_L_tensor_border(site-1)

                            if self.get_use_orthogonal_MPSs():
                                self._set_L_tensor_border_orthogonal(site-1)

                            (self.system.
                             move_right_normalized_boundary_to_the_left())

                            self.system.set_R_tensor_border(site)

                            if self.get_use_orthogonal_MPSs():
                                self._set_R_tensor_border_orthogonal(site)

                    else:

                        if site < length - 1:
                            self.system.set_R_tensor_border(site+2)

                            if self.get_use_orthogonal_MPSs():
                                self._set_R_tensor_border_orthogonal(site+2)

                            (self.system.
                             move_left_normalized_boundary_to_the_right())

                            self.system.set_L_tensor_border(site+1)

                            if self.get_use_orthogonal_MPSs():
                                self._set_L_tensor_border_orthogonal(site+1)

                # print progress if wanted
                if print_steps:
                    self._print_sweep_status(sweep,sweeps,direction,site,sites,
                                             length,
                                             result_array[0,sweep-1],
                                             result_array[1,sweep-1],
                                             result_array[2,sweep-1])

            # update energy and energy variance in results_array
            energy_variance, energy = self.system.energy_variance()
            result_array[0,sweep] = np.real(energy)
            result_array[1,sweep] = np.real(energy_variance)

            # reverse sweep direction
            direction = 'L' if direction == 'R' else 'R'

        # return results_array
        return result_array

    def modify_effective_Hamiltonian2(self,H,site,two_site,
                                      fast=False,Delta=10):
        '''
        Modifies the given effective Hamiltonian by elevating the energies of
        the MPSs this MPS should be held orthogonal to by 'Delta'. If the
        states stored in these MPSs lie below the target state in terms of
        energy by a difference smaller than 'Delta', the target state will
        become the ground state of the modified Hamiltonian under the
        assumption that no other state lie below the target state thus enabling
        its determination by the ground state DMRG.

        This function accepts the following arguments:

        H        : The current effective Hamiltonian that has not yet been
                   modified. May be of type np.ndarray or an instance of the
                   classes DMRG_LO.
        site     : The position in the spin chain the effective Hamiltonian H
                   refers to. For two-site DMRG, the left site has to be
                   selected.
        two_site : Whether two-site DMRG should be used or one-site DMRG.
        fast     : Whether the chi^3 scaling should be used or chi^4 scaling.
                   Default is False. This argument must correspond to the
                   datatype of H. fast=False for H being of type np.ndarray and
                   fast=True for H being an instance of class DMRG_LO.
        Delta    : The amount by which the energies of the states this MPS
                   should be held orthogonal to are elevated. Default is 10.

        If fast=False and the numbers stored in the np.ndarray representing H
        are of type np.float32, np.float64, np.complex64 or np.complex128, the
        modification of H will be done inplace to avoid unnecessary memory
        allocations.

        This function returns the modified effective Hamiltonian. The datatype
        is based on the datatype of the unmodified effective Hamiltonian 'H'.
        If H is of type np.ndarray, the returned object is also if type
        np.ndarray. However, if H is an instance of class DMRG_LO or
        DMRG_LO_REDUCED, the returned object will be an instance of class
        DMRG_LO_PROJECTION2.

        TODO: This function does not work with for H being a reduced tensor. It
              also does not work for H being of class DMRG_LO_REDUCED.
        '''

        a = len(self.orthogonal_MPSs)

        # Prepare orthogonal environment
        for i in range(a):

            # index order: c_p-1 | a_p-1
            L = self._L_tensors_orthogonal[i][-1]

            # one-site DMRG index order: c_p | a_p
            # two-site DMRG index order: c_p+1 | a_p+1
            R = self._R_tensors_orthogonal[i][-1]

            if two_site:

                # index order: s_p | s_p+1 | c_p+1 | c_p-1
                M = self.system._dagger(
                    self.orthogonal_MPSs[i].get_twosite_tensor(site))

                # index order: s_p | s_p+1 | c_p+1 | c_p-1 @ c_p-1 | a_p-1
                #           => s_p | s_p+1 | c_p+1 | a_p-1
                M = tensordot(M,L,axes=((3),(0)))

                # index order: s_p | s_p+1 | c_p+1 | a_p-1 @ c_p+1 | a_p+1
                #           => s_p | s_p+1 | a_p-1 | a_p+1
                M = tensordot(M,R,axes=((2),(0)))

            else:

                # index order: s_p | c_p | c_p-1
                M = self.system._dagger(self.orthogonal_MPSs[i].get_site(site))

                # index order: s_p | c_p | c_p-1 @ c_p-1 | a_p-1
                #           => s_p | c_p | a_p-1
                M = tensordot(M,L,axes=((2),(0)))

                # index order: s_p | c_p | a_p-1 @ c_p | a_p
                #           => s_p | a_p-1 | a_p
                M = tensordot(M,R,axes=((1),(0)))

            if i == 0 and a > 1:
                N = np.empty([*M.shape,a])
            elif a == 1:
                N = M
                break

            N[...,i] = M

        if fast:

            # return an instance of DMRG_LO_PROJECTION2
            # to which most of the work is then delegated to.
            return DMRG_LO_PROJECTION2(H,N,Delta)
        else:

            # find out suitable BLAS subroutine
            sger = None
            if H.dtype == np.float32:
                sger = sp.linalg.blas.sger
            elif H.dtype == np.float64:
                sger = sp.linalg.blas.dger
            elif H.dtype == np.complex64:
                sger = sp.linalg.blas.cgerc
            elif H.dtype == np.complex128:
                sger = sp.linalg.blas.zgerc

            # Update H inplace if possible
            if sger is not None and H.flags.f_contiguous:
                sger(alpha=Delta,a=H,x=N,y=N,overwrite_a=1,
                     overwrite_x=0,overwrite_y=0)
            else:
                H += Delta*np.outer(N,np.conj(N))

            return H

    def modify_effective_Hamiltonian(self,H,site,two_site,fast=False):
        '''
        Modifies the given effective Hamiltonian by setting the energies of the
        MPSs this MPS should be held orthogonal to to zero. If the states
        stored in these MPSs lie below the target state in terms of energy by a
        difference smaller than 'Delta', the target state will become the
        ground state of the modified Hamiltonian and thus enabling its
        determination by the ground state DMRG.

        If the target state has a negative energy and the states stored in the
        MPS this MPS should be held orthogonal to are the only eigenstates
        below the target state, the target state will become the ground state
        of the modified Hamiltonian thus enabling its determination by the
        ground state DMRG.                           

        This function accepts the following arguments:

        H        : The current effective Hamiltonian that has not yet been
                   modified. May be of type np.ndarray or an instance of the
                   classes DMRG_LO.
        site     : The position in the spin chain the effective Hamiltonian H
                   refers to. For two-site DMRG, the left site has to be
                   selected.
        two_site : Whether two-site DMRG should be used or one-site DMRG.
        fast     : Whether the chi^3 scaling should be used or chi^4 scaling.
                   Default is False. This argument must correspond to the
                   datatype of H. fast=False for H being of type np.ndarray and
                   fast=True for H being an instance of class DMRG_LO.

        If fast=False, the number of orthogonal MPSs is one and the numbers
        stored in the np.ndarray representing H are of type np.float32,
        np.float64, np.complex64 or np.complex128, the modification of H will
        be done inplace to avoid unnecessary memory allocations.

        This function returns the modified effective Hamiltonian. The datatype
        is based on the datatype of the unmodified effective Hamiltonian 'H'.
        If H is of type np.ndarray, the returned object is also if type
        np.ndarray. However, if H is an instance of class DMRG_LO or
        DMRG_LO_REDUCED, the returned object will be an instance of class
        DMRG_LO_PROJECTION.

        TODO: This function does not work with for H being a reduced tensor. It
              also does not work for H being of class DMRG_LO_REDUCED.
        '''

        a = len(self.orthogonal_MPSs)

        # Prepare orthogonal environment
        for i in range(a):

            # index order: c_p-1 | a_p-1
            L = self._L_tensors_orthogonal[i][-1]

            # one-site DMRG index order: c_p | a_p
            # two-site DMRG index order: c_p+1 | a_p+1
            R = self._R_tensors_orthogonal[i][-1]

            if two_site:

                # index order: s_p | s_p+1 | c_p+1 | c_p-1
                M = self.system._dagger(
                    self.orthogonal_MPSs[i].get_twosite_tensor(site))

                # index order: s_p | s_p+1 | c_p+1 | c_p-1 @ c_p-1 | a_p-1
                #           => s_p | s_p+1 | c_p+1 | a_p-1
                M = tensordot(M,L,axes=((3),(0)))

                # index order: s_p | s_p+1 | c_p+1 | a_p-1 @ c_p+1 | a_p+1
                #           => s_p | s_p+1 | a_p-1 | a_p+1
                M = tensordot(M,R,axes=((2),(0)))

            else:

                # index order: s_p | c_p | c_p-1
                M = self.system._dagger(self.orthogonal_MPSs[i].get_site(site))

                # index order: s_p | c_p | c_p-1 @ c_p-1 | a_p-1
                #           => s_p | c_p | a_p-1
                M = tensordot(M,L,axes=((2),(0)))

                # index order: s_p | c_p | a_p-1 @ c_p | a_p
                #           => s_p | a_p-1 | a_p
                M = tensordot(M,R,axes=((1),(0)))

            if i == 0 and a > 1:
                N = np.empty([*M.shape,a])
            elif a == 1:
                N = M
                break

            N[...,i] = M

        if fast:

            # return an instance of DMRG_LO_PROJECTION
            # to which most of the work is then delegated to.
            return DMRG_LO_PROJECTION(H,N)
        else:

            # for dense H in (n x n), optimised for speed,
            # may need to create an aux. array of size(H).
            # Flops: 6 a n^2 + 2 n^2 , n = D^2 * d^x

            # Project N out
            if a == 1:
                N = N.reshape(np.prod(N.shape))
            else:
                N = N.reshape(np.prod(N.shape[:-1]),N.shape[-1])
            A = -np.tensordot(H,N,axes=((1),(0)))

            if a == 1:

                # performs in 6n^2.
                # 2*n^2 for A
                # 2*n^2 for first outer product update
                # 2*n^2 for second outer product update

                # find out suitable BLAS subroutine
                sger = None
                if H.dtype == np.float32:
                    sger = sp.linalg.blas.sger
                elif H.dtype == np.float64:
                    sger = sp.linalg.blas.dger
                elif H.dtype == np.complex64:
                    sger = sp.linalg.blas.cgerc
                elif H.dtype == np.complex128:
                    sger = sp.linalg.blas.zgerc

                # Update H inplace if possible
                if sger is not None and H.flags.f_contiguous:

                    sger(alpha=1,a=H,x=A,y=N,overwrite_a=1,
                         overwrite_x=0,overwrite_y=0)
                    A = -np.tensordot(N,H,axes=((0),(0)))
                    sger(alpha=1,a=H,x=N,y=A,overwrite_a=1,
                         overwrite_x=0,overwrite_y=0)

                else:

                    # manual implementation
                    H += np.outer(A,np.conj(N))
                    A = -np.tensordot(N,H,axes=((0),(0)))
                    H += np.outer(N,np.conj(A))
            else:

                # manual implementation
                H += np.dot(A,N.T)
                A = -np.tensordot(N,H,axes=((0),(0)))
                H += np.dot(N,A.T)

        return H

    def _optimisation_step(self,H,v0,method='g'):
        '''
        The optimisation step in which an eigenstate of the effective
        Hamiltonian 'H' is calculated according to the supplied method. Based
        on the value of 'method', the argument v0 serves either as an initial
        guess, influences the optimisation in another capacity or does not
        influence it at all.

        This function takes the following arguments:
        
        H      : The effective Hamiltonian. May be of type np.ndarray or
                 rt.reducedTensor or an instance of the classes DMRG_LO,
                 DMRG_LO_REDUCED, DMRG_LO_PROJECTION, DMRG_LO_PROJECTION2.
                 The index order must be (s'_k|a'_k-1| a'_k)|(s_k|a_k-1|a_k)
                 for one-site DMRG and
                 (s'_k|s'_k+1|a'_k-1| a'_k)|(s_k|s_k+1|a_k-1|a_k)
                 for two-site DMRG.
        v0     : The tensor accompanying the effective Hamiltonian. Its type
                 must correspond with the type of H. If H is of type
                 np.ndarray, DMRG_LO, DMRG_LO_PROJECTION or
                 DMRG_LO_PROJECTION2, v0 must be of type np.ndarray. If H is of
                 type rt.reducedTensor or
                 DMRG_LO_REDUCED, v0 must be of type rt.reducedTensor. The
                 index order of v0 must be (s_k|a_k-1|a_k) for one-site DMRG
                 and (s_k|s_k+1|a_k-1|a_k) for two-site DMRG.
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
        '''

        if method == 'g' or method == 'h':
            # perform DMRG

            # reshape v0 into a vector
            dimM = np.shape(v0)
            if self.system._useReducedTensors:
                if type(H) != DMRG_LO_REDUCED:
                    rt.combine_axes_for_reduced_tensor(v0,+1,0,len(dimM))
            else:
                v0 = v0.reshape(np.prod(dimM))

            which = 'SA' if method == 'g' else 'LA'

            if self.system._useReducedTensors:

                # for reduced Tensors, delegate everything to the
                # respective functions
                if type(H) == DMRG_LO_REDUCED:
                    v0 = H.cut_dense_part_out_M(v0)
                    w,v = linalg.eigsh(A=H, v0=v0, k=1, which=which,
                                       tol=self.sweep_tolerance, sigma=None)
                    v = H.slice_dense_part_in_sparse_M(v)
                    return w,v
                else:
                    w,v = rt.groundstate_searcher_for_reduced_matrix(H,v0,
                                                                 method=method)
                    rt.split_axis_for_reduced_tensor(v,0)
                    return w,v

            else:

                # for dense Tensors, use scipy.linalg.eigsh
                # to calculate the ground state.

                # Try to calculate eigenstate.
                try:
                    w,v = linalg.eigsh(A=H,v0=v0,k=1,which=which,
                                       tol=self.sweep_tolerance,sigma=None)
                    failure = False
                except sp.sparse.linalg.ArpackNoConvergence:
                    failure = True

                # If failed, try again with more Lanczos vectors (40).
                if failure:
                    print('ARPACK eigensolver did not converge. '
                          'Trying again with 40 Lanczos vectors.')
                    try:
                        w,v = linalg.eigsh(A=H,v0=v0,k=1,which=which,
                                           tol=self.sweep_tolerance,
                                           sigma=None,ncv=40)
                        failure = False
                    except sp.sparse.linalg.ArpackNoConvergence:
                        pass # so that other errors get raised

                # If failed, try again with more Lanczos vectors (80).
                # If failed again, raise an error.
                if failure:
                    print('ARPACK eigensolver did not converge. '
                          'Trying again with 80 Lanczos vectors.')
                    try:
                        w,v = linalg.eigsh(A=H,v0=v0,k=1,which=which,
                                           tol=self.sweep_tolerance,
                                           sigma=None,ncv=80)
                        failure = False
                    except sp.sparse.linalg.ArpackNoConvergence as e:
                        raise e

                return w[0],v.reshape(dimM)


        elif method == 'gED' or method == 'hED':
            # the same as 'g' and 'h' but an exact eigensolver is used
            # and not an iterative one. Here, v0 will remain unused.

            dimM = np.shape(v0)

            if self.system._useReducedTensors:

                # For reduced tensors, delegate everything to the corresponding
                # function from the module rt.
                rt.combine_axes_for_reduced_tensor(v0,+1,0,len(dimM))
                w,v = rt.groundstate_searcher_for_reduced_matrix(H,v0,
                                                                 method=method,
                                                              save_memory=True)
                rt.split_axis_for_reduced_tensor(v,0)
                return w,v

            # For dense tensors, use scipy.linalg.eigh.
            w,v = sp.linalg.eigh(a=H,overwrite_a=True,eigvals=None,
                                 check_finite=False)

            if method == 'gED':
                return w[0],v[:,0].reshape(dimM)
            else:
                return w[-1],v[:,-1].reshape(dimM)


        elif method == 'DMRGX':
            # perform DMRG-X

            dimM = np.shape(v0)

            # For reduced tensors, delegate everything to the corresponding
            # function from the module rt.
            if self.system._useReducedTensors:
                rt.combine_axes_for_reduced_tensor(v0,+1,0,len(dimM))
                w,v = rt.groundstate_searcher_for_reduced_matrix(H,v0,
                                                                 method=method,
                                                              save_memory=True)
                rt.split_axis_for_reduced_tensor(v,0)
                return w,v

            # For dense tensors, use scipy.linalg.eigh.
            w,v = sp.linalg.eigh(a=H,overwrite_a=True,eigvals=None,
                                 check_finite=False)

            # find the state with the highest overlap to the previous one
            v0 = np.conjugate(v0.reshape(np.prod(dimM)))
            overlap = np.abs(tensordot(v0,v,axes=[[0],[0]]))
            highest_overlap_index = np.argmax(overlap)

            return (w[highest_overlap_index],
                    v[:,highest_overlap_index].reshape(dimM))

        else:
            raise ValueError("The method '{}' is not ".format(method)+
                             "recognised. Choose from the following list: "
                             "['g','h','gED','hED','DMRGX']")

    def _print_sweep_status(self,sweep,sweeps,direction,site,sites,
                            length,E,DeltaE,truncErr):
        '''
        Prints the status of the simulation once a site optimisation has been
        finished by showing the information provided to this function. To do
        so, the information already printed to the active line is overwritten
        thus providing dynamic output capabilities. Each time this function
        prints a line, a newline character (\n) is omited. During the very last
        printout, a newline character is included to finish the active line.

        This function accepts the following arguments:

        sweep     : The current sweep that is being conducted.
        sweeps    : The number of sweeps ordered in total.
        direction : The direction toward which the current sweep sweeps.
        site      : The current site being optimised. If more than one site is
                    optimised, must give the position of the left most site
                    involved.
        sites     : The number of sites being optimised.
        length    : The length of the MPS.
        E         : The current energy of the MPS.
        DeltaE    : The current energy vairance of the MPS.
        truncErr  : The truncation error received after the last compression.
        '''

        # the character representing the currently active sites
        bar_char = '<' if direction == 'L' else '>'

        # shorten output if MPS is longer than 50 sites
        if length > 50:
            ratio = length/20
            length_here = 20
            site_here   = int(site/ratio)
        else:
            length_here = length
            site_here   = site

        # the string representing the inactive sites left and right
        # of the active ones
        bar_left  = (site_here*' ' if sites == 1 or site < length-3
                     else (site_here-1)*' ')
        bar_right = (length_here-site_here-sites)*' '

        l = int(np.log10(length))+1

        # print output
        if sweep == 0:
            print('\r[{}{}{}] site: {}/{}, sweep: {}/{}, dir.: {}'.format(
                bar_left,sites*bar_char,bar_right,
                str(site+1).rjust(l),length,sweep+1,sweeps,direction),end='')
        else:
            if sites == 1:
                print('\r[{}{}{}] '.format(bar_left,bar_char,bar_right)+
                      'site: {}/{}, '.format(str(site+1).rjust(l),length)+
                      'sweep: {}/{}, '.format(sweep+1,sweeps)+
                      'dir.: {}, E: {:12.5e}, Delta E: {:9.2e}  '.format(
                          direction,E,DeltaE),end='')

            else:
                print('\r[{}{}{}] '.format(bar_left,sites*bar_char,bar_right)+
                      'site: {}/{}, '.format(str(site+1).rjust(l),length)+
                      'sweep: {}/{}, '.format(sweep+1,sweeps)+
                      'dir.: {}, E: {:12.5e}, '.format(direction,E)+
                      'Delta E: {:9.2e}, truncErr: {:8.2e}   '.format(
                          DeltaE,truncErr),end='')

        # print newline character if the calculation is finished
        # to properly end the line.
        if (sweep+1 == sweeps and ((direction=='R' and sites+site==length) or
                                   (direction=='L' and site==0))):
            print()
