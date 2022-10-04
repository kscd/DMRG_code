import numpy as np
import scipy as sp
import copy
import scipy.sparse.linalg as linalg
import MPS

# import reducedTensor class if it exists
# if not, only dense tensors can be used
try:
    import reducedTensor as rt
    _reducedTensorExists = True
except ImportError:
    _reducedTensorExists = False


class tMPS:
    '''
    The class 'tMPS' is a class performing the tMPS calculation and holding all
    necessary data for it. In this sense, it is an abstract class because it
    does not represent something one can 'touch'. The class has the ability to
    perform first order, second order or fourth order trotterization and has
    the ability to run an evaluation function given to it. Furthermore, it is
    limited to Hamiltonians with nearest-neigbor interactions but can deal with
    time-dependent as well as time-independent Hamiltonians. When the time
    evolution is performed, it is possible analyse the MPS by providing an
    evaluation function. This class provides full support for MPSs where the
    site tensors are either of type np.ndarray or of type rt.reducedTensor.
    This class is only able to work on instances of MPS_system where open
    boundary conditions are used.
    '''

    def __init__(self,MPS,f,Delta_t=0.01,t0=0,*,operating_distance=2,
                 f_args_list=[],f_kwargs_dict={},
                 trotter_level=1,use_U1_symmetry=False):
        '''
        Initialises the tMPS class. This is done by providing the MPS which is
        to be time-evolved and an operator chain that describes the
        Hamiltonian under which the time evolution occurs among other numeric
        parameters.

        For a one-dimensional spin system consisting of L spin sites, an
        operator chain is a list of length L-1 each holding the two-site
        interaction term between two neighboring sites as a matrix of shape
        (d**2,d**2) and type np.ndarray. The first entry describes how the
        first and second site interact, the second entry how the second and
        third site interact and so forth. The index order of the k-th entry in
        the operator chain is: (s'_k | s'_k+1) | (s_k | s_k+1) if k counts
        from one.

        For time-independent Hamiltonians, the argument 'f' provides operator
        strings directly as a list, while for time-independent Hamiltonians,
        the argument 'f' is a function returning the operator chain at a given
        time upon being called.

        This function takes the following arguments:

        MPS                : The MPS to be time-evolved. Instance of class MPS.
        f                  : This argument describes the Hamiltonian under
                             which the time evolution is performed. For the
                             case of a time-independent Hamiltonian, it is a
                             list containing the operator chains directly,
                             while for time-dependent Hamiltonians it is a
                             function that returns the operator chains and
                             accepts as first argument the time and as further
                             arguments the arguments provided in f_args_list
                             and f_kwargs_dict.
        Delta_t            : The time interval with which the MPS is
                             time-evolved. Default is 0.01.
        t0                 : The beginning of time. Default is 0.
        operating_distance : The spatial range of the interaction terms of the
                             Hamiltonian. For now only 2.
        f_args_list        : A list of position arguments for the function f.
                             Default is [] indicating no additional arguments.
                             Is ignored for time-independent Hamiltonians.
        f_kwargs_dict      : A dictionary containing the keyword arguments for
                             the function f. Default is {} indicating no
                             additional arguments. Is ignored for
                             time-independent Hamiltonians.
        trotter_level      : The trotter level to be used. Possible choices are
                             1, 2 and 4. Default is 1.
        use_U1_symmetry    : Whether to use reduced tensor or not. Has to
                             correspond with the MPS in question. Default is
                             False.
        '''

        # check operation distance
        if operating_distance == 2:
            self._operating_distance = 2
        else:
            raise ValueError('Only 2 body interactions '
                             'are supported right now.')

        # set attributes
        self.MPS = MPS
        self._d = MPS._d
        self.Delta_t = Delta_t

        self._t = t0

        self._f = f
        self._f_args = f_args_list
        self._f_kwargs = f_kwargs_dict

        self._time_dependent = callable(f)
        self.use_U1_symmetry = use_U1_symmetry
        self.trotter_level = trotter_level

        if not self._time_dependent:

            # get the operator chain now and re-calculate it never
            self._time_evolution_MPOs = self._create_time_evolution_MPOs_2site(
                f,Delta_t,trotter_level)

    def _operator_to_local_MPO_2site(self,operator):
        '''
        Takes an operator of type np.ndarray acting on two sites as a matrix of
        shape (d**2,d**2) and decomposes it into an MPO with two sites. If U(1)
        symmetries are used, the resulting MPO sites are reducedTensors and
        numpy arrays if not. The index order of the given operator is:
        (s'_p | s'_p+1) | (s_p | s_p+1). Returns the tuple (U,V) with U
        describing the first MPO site tensor and V describing the second MPO
        site tensor. The index orders of U and V as well as the shapes are as
        follows with 'd' being the local state space dimension:
        U | b_p-1 | b_p | s'_p | s_p     | (1,d,d,d)
        V | b_p | b_p+1 | s'_p+1 | s_p+1 | (d,1,d,d)
        '''

        d = self._d

        # index order: (s'_p | s'_p+1) | (s_p | s_p+1)
        #           =>  s'_p | s'_p+1  |  s_p | s_p+1
        operator = operator.reshape([d,d,d,d])

        # index order: s'_p | s'_p+1  |  s_p | s_p+1
        #           => s'_p | s_p  |  s'_p+1 | s_p+1
        operator = operator.swapaxes(1,2)

        if self.use_U1_symmetry:

            # use reduced tensors

            operator = rt.reducedTensor(tensor=operator,
                                        list_of_q=[[-1,1],[-1,1],
                                                   [-1,1],[-1,1]],
                                        list_of_xi=[1,-1,1,-1])

            # index order: s'_p | s_p |  s'_p+1 | s_p+1
            #           => s'_p | s_p | (s'_p+1 | s_p+1)
            q1,s1 = rt.combine_axes_for_reduced_tensor(operator,+1,2,2)

            # index order:  s'_p | s_p  | (s'_p+1 | s_p+1)
            #           => (s'_p | s_p) | (s'_p+1 | s_p+1)
            q2,s2 = rt.combine_axes_for_reduced_tensor(operator,-1,0,2)

            # U index order: (s'_p | s_p) | b_p
            # V index order: b_p | (s'_p+1 | s_p+1)
            U,_,V,t = rt.SVD_for_reduced_matrix(operator,normalize='n')

            # index order: (s'_p | s_p) | b_p => s'_p | s_p | b_p
            rt.split_axis_for_reduced_tensor(U,0,q2,s2)

            # index order: b_p | (s'_p+1 | s_p+1) => b_p | s'_p+1 | s_p+1
            rt.split_axis_for_reduced_tensor(V,1,q1,s1)

            # index order: s'_p | s_p | b_p => s'_p | s_p | b_p-1 | b_p
            rt.insert_dummy_dimension(U,2,-1)

            # index order: b_p | s'_p+1 | s_p+1 => b_p | b_p+1 | s'_p+1 | s_p+1
            rt.insert_dummy_dimension(V,1,1)

            # index order: s'_p | s_p | b_p-1 | b_p => b_p-1 | s_p | s'_p | b_p
            U = rt.swapaxes_for_reduced_tensor(U,0,2)

            # index order: b_p-1 | s_p | s'_p | b_p => b_p-1 | b_p | s'_p | s_p
            U = rt.swapaxes_for_reduced_tensor(U,1,3)

            return U,V

        else:

            # use dense tensors

            # index order:  s'_p | s_p  |  s'_p+1 | s_p+1
            #           => (s'_p | s_p) | (s'_p+1 | s_p+1)
            operator = operator.reshape([d**2,d**2])

            shape = np.shape(operator)

            # U index order: (s'_p | s_p) | b_p
            # S index order: b_p | b_p
            # V index order: b_p | (s'_p+1 | s_p+1)
            U,S,V = np.linalg.svd(operator)

            # index order: (s'_p | s_p) | b_p @ b_p | b_p => (s'_p | s_p) | b_p
            U = np.dot(U,np.diag(np.sqrt(S)))

            # index order: b_p | b_p @ b_p | (s'_p+1 | s_p+1)
            #           => b_p | (s'_p+1 | s_p+1)
            V = np.dot(np.diag(np.sqrt(S)),V)

            # index order: (s'_p | s_p) | b_p => s'_p | s_p | b_p-1 | b_p
            U = U.reshape(d,d,1,d**2)

            # index order: b_p | (s'_p+1 | s_p+1)
            #           => b_p | b_p+1 | s'_p+1 | s_p+1
            V = V.reshape(d**2,1,d,d)

            # index order: s'_p | s_p | b_p-1 | b_p => b_p-1 | s_p | s'_p | b_p
            U = U.swapaxes(0,2)

            # index order: b_p-1 | s_p | s'_p | b_p => b_p-1 | b_p | s'_p | s_p
            U = U.swapaxes(1,3)

            return U,V

    def _create_time_evolution_MPOs_2site(self,operator_chain,Delta_t,
                                          trotter_level=1):
        '''
        This function creates the short two-site MPOs used in the time
        evolution. These MPOs are then applied to the MPS in order to move the
        bond they describe forward the amound of time specified in 'Delta_t. 
        The MPOs are created out of the given operator chain and one two-site
        MPO represents one operator in the chain. By combining all two-site
        MPOs describing either the even or the odd bonds, a full-length MPO is
        created with a local bond dimension alternating between 1 and d with
        'd' being the local state space dimension. A dummy MPO site tensor is
        inserted at the edges whenever necessary.

        This function returns a list with its entries being such full-length
        MPOs. The exact number and nature of the returned MPOs depends on the
        used trotterization order 'trotter_level'. The possible values are the
        following, where the status of the time-dependency has been provided in
        the initialisation of this class:

        trotter_level=1: 2 MPOs returned.
        1. MPO describes all even bonds, moves forward full time step.
        2. MPO describes all odd bonds, moves forward full time step.

        trotter_level=2 and time dependent: 2 MPOs returned.
        1. MPO describes all even bonds, moves forward half time step.
        2. MPO describes all odd bonds, moves forward full time step.

        trotter_level=2 and time independent: 3 MPOs returned.
        1. MPO describes all even bonds, moves forward half time step.
        2. MPO describes all even bonds, moves forward full time step.
        3. MPO describes all odd bonds, moves forward full time step.

        trotter_level=4: 5 MPOs returned.
        1. MPO describes all even bonds, moves forward t_1 / 2 step.
        2. MPO describes all even bonds, moves forward t_1 step.
        3. MPO describes all even bonds, moves forward (t_1 + t_3)/2 step.
        4. MPO describes all odd bonds, moves forward t_1 step.
        5. MPO describes all odd bonds, moves forward t_3 step.

        with t_1 = 1/( 4 - np.power(4,1/3) ) * Delta_t
        and  t_3 = Delta_t - 4*t_1
        '''

        d = self._d

        # create dummy MPO site needed at the edges
        dummy_MPO_site = np.empty([1,1,d,d])
        dummy_MPO_site[0,0] = np.eye(d)

        if self.use_U1_symmetry:
            dummy_MPO_site = rt.reducedTensor(dummy_MPO_site,
                                              list_of_q = [[0],[0],
                                                           [-1,1],[-1,1]],
                                              list_of_xi = [-1,1,1,-1])

        # create two-site MPOs depending on trotterization order
        if trotter_level == 1:
            # e o | e o | e o | e o
            # e in 0
            # o in 1

            # initialise MPO chains, insert dummy sites at the beginning
            MPO_chains = [[],[dummy_MPO_site]]

            # create full-length MPOs out of short two-site MPOs
            for index,h in enumerate(operator_chain):
                eh = sp.linalg.expm(-1j*Delta_t*h)
                miniMPO = self._operator_to_local_MPO_2site(eh)
                MPO_chains[index % 2].extend(miniMPO)

            # insert dummy sites at the ends and return
            MPO_chains[1 - self.MPS.get_length() % 2].append(dummy_MPO_site)
            return MPO_chains

        elif trotter_level == 2 and self._time_dependent:
            # e/2 o e/2 | e/2 o e/2 | e/2 o e/2 | e/2 o e/2
            # e/2 in 0
            # o   in 1
            
            # initialise MPO chains, insert dummy sites at the beginning
            MPO_chains = [[],[dummy_MPO_site]]

            # create full-length MPOs out of short two-site MPOs
            for index,h in enumerate(operator_chain):
                factor = 0.5 if index % 2 == 0 else 1
                eh = sp.linalg.expm(-1j*Delta_t*h*factor)
                miniMPO = list(self._operator_to_local_MPO_2site(eh))
                MPO_chains[index % 2].extend(miniMPO)

            # insert dummy sites at the ends and return
            MPO_chains[1 - self.MPS.get_length() % 2].append(dummy_MPO_site)
            return MPO_chains

        elif trotter_level == 2 and not self._time_dependent:
            # e/2 o e o e o e o e/2
            # e/2 in 0
            # e   in 1
            # o   in 2

            # initialise MPO chains, insert dummy sites at the beginning
            MPO_chains = [[],[],[dummy_MPO_site]]

            # create full-length MPOs out of short two-site MPOs
            for index,h in enumerate(operator_chain):

                if index % 2 == 0:

                    # even sites

                    eh = sp.linalg.expm(-1j*Delta_t*h/2)
                    miniMPO = list(self._operator_to_local_MPO_2site(eh))
                    MPO_chains[0].extend(miniMPO)

                    eh = sp.linalg.expm(-1j*Delta_t*h)
                    miniMPO = list(self._operator_to_local_MPO_2site(eh))
                    MPO_chains[1].extend(miniMPO)

                else:

                    # odd sites

                    eh = sp.linalg.expm(-1j*Delta_t*h)
                    miniMPO = list(self._operator_to_local_MPO_2site(eh))
                    MPO_chains[2].extend(miniMPO)

            # insert dummy sites at the ends and return
            if self.MPS.get_length() % 2 == 0:
                MPO_chains[2].append(dummy_MPO_site)
            else:
                MPO_chains[0].append(dummy_MPO_site)
                MPO_chains[1].append(dummy_MPO_site)

            return MPO_chains

        elif trotter_level == 4:
            # 0 e(t1/2)
            # 1 e(t1)
            # 2 e((t1+t3)/2)
            # 3 o(t1)
            # 4 o(t3)

            tau_1 = 1/( 4 - np.power(4,1/3) ) * Delta_t
            tau_3 = Delta_t - 4*tau_1

            # initialise MPO chains, insert dummy sites at the beginning
            MPO_chains = [[],[],[],[dummy_MPO_site],[dummy_MPO_site]]

            # create full-length MPOs out of short two-site MPOs
            for index,h in enumerate(operator_chain):

                if index % 2 == 0:

                    # even sites

                    eh = sp.linalg.expm(-1j*tau_1*h/2)
                    miniMPO = list(self._operator_to_local_MPO_2site(eh))
                    MPO_chains[0].extend(miniMPO)

                    eh = sp.linalg.expm(-1j*tau_1*h)
                    miniMPO = list(self._operator_to_local_MPO_2site(eh))
                    MPO_chains[1].extend(miniMPO)

                    eh = sp.linalg.expm(-1j*(tau_1+tau_3)*h/2)
                    miniMPO = list(self._operator_to_local_MPO_2site(eh))
                    MPO_chains[2].extend(miniMPO)

                else:

                    # odd sites

                    eh = sp.linalg.expm(-1j*tau_1*h)
                    miniMPO = list(self._operator_to_local_MPO_2site(eh))
                    MPO_chains[3].extend(miniMPO)

                    eh = sp.linalg.expm(-1j*tau_3*h)
                    miniMPO = list(self._operator_to_local_MPO_2site(eh))
                    MPO_chains[4].extend(miniMPO)

            # insert dummy sites at the ends and return
            if self.MPS.get_length() % 2 == 0:
                MPO_chains[3].append(dummy_MPO_site)
                MPO_chains[4].append(dummy_MPO_site)
            else:
                MPO_chains[0].append(dummy_MPO_site)
                MPO_chains[1].append(dummy_MPO_site)
                MPO_chains[2].append(dummy_MPO_site)

            return MPO_chains


    def _compress_MPS(self,start,D=None,max_truncation_error=None):
        '''
        Internal function which compresses the MPS by compressing the even
        (start = 0) or the odd (start = 1) bonds. It is used by the function
        'evolve' after each MPS-MPO multiplication and the new local bond
        dimension can be either set directly by 'D' or indirectly by the
        maximum truncation error the compression is willing to accept. If both
        'D' and 'max_truncation_error' are set, the value given by 'D' acts as
        a hard limit overrulling the provided truncation error if necessary.
        '''

        even_bonds = True if start == 0 else False
        tr_err = self.MPS.compress_MPS_with_SVD(D,max_truncation_error,
                                                False,even_bonds)
        return tr_err


    def evolve(self,time_steps,Delta_t=None,D=None,max_truncation_error=1e-14,
               evaluation_function=None,evaluation_step=-1,
               evaluation_args_list=[],evaluation_kwargs_dict={},
               trotter_level=1,abort_at_max_D=False,
               *,print_info=False):
        '''
        Performs the time evolution algorithm tMPS on the instance of the class
        MPS stored in the present instance of this class for a number of time
        steps, each advancing the system a given time. To do this, it is
        possible to use first, second or fourth order trotterization. It is
        assumed that the Hamiltonian used in this endeavor is already known to
        the present instance.

        For time-independent Hamiltonians and trotterization orders larger than
        one, it is possible to speed the calculation up by merging the last
        calculation from one time step and the first calculation of the next
        time step into one. Here, this possibility is only available to second
        order trotterization where it saves up to 33.33% of the work and takes
        the form: (e/2 o e/2) (e/2 o e/2) -> e/2 o e o e/2 in the example where
        of two time steps. For fourth order trotterization this feature is not
        available as the saved time lies is merely up to 10%. Merging time
        steps like this means that the MPS never describes the state of the
        system at intermediate times.

        This function offers the capability of accepting a function, the
        evaluation function, that is executed every few time evolution steps.
        Such a function is most suitable to measure, for example, observables
        in dependence of time. To perform these measurements, the MPS is
        brought into a valid form to correctly describe the state of the system
        after the respective number of time steps has been performed.
        Additional arguments that may be given to this function are supplied by
        the respective arguments to this function.

        This function returns:
        (trunc_err, D_max) if evaluation_function=None or evaluation_step=-1,
        (trunc_err, D_max, evaluation_results) otherwise.
        Here, 'trunc_err' is the truncation error and 'D_max' the largest local
        bond dimension found in the MPS after each time step. For second order
        trotterization and time-indenpendent Hamiltonian, 'trunc_err' and
        'D_max' refer to the next half time step. In 'evaluation_results', the
        value returned by the evaluation function is stored in dependence of
        time.

        This function takes the following arguments:

        time_steps             : The number of time evolution steps to be
                                 performed.
        Delta_t                : The step in time, each time step takes
                                 forward. If set to None, fallback to the value
                                 provided in the initialisation. Default is
                                 None.
        D                      : The local bond dimension the MPS should be
                                 kept at during the calculation. Setting it to
                                 None results in a possibly unbounded growth of
                                 the local bond dimension.
        max_truncation_error   : The truncation error that is accepted during
                                 the compression of the MPS after each substep.
                                 Numerical values are not comparable across
                                 trotter levels. May be overruled by 'D'
                                 depending on the value of 'abort_at_max_D'.
                                 Default is 1e-14.
        evaluation_function    : The function to be evalued every
                                 'evaluation_step' steps. To call this
                                 function, the tMPS algorithm needs to bring
                                 the MPS in valid form which thus inducing a
                                 possible overhead. The first argument it
                                 accepts must be the MPS given to it.
                                 Additional arguments may be provided
                                 thereafter. Default is None, indicating that
                                 no analysis should be performed.
        evaluation_step        : The number of time steps between calls to the
                                 evaluation function. Default is -1 indicating
                                 that the given evaluation function is never
                                 called.
        evaluation_args_list   : A list of positional arguments given to the
                                 evaluation function. Default is [] indicating
                                 that no arguments are given.
        evaluation_kwargs_dict : A dictionary of keyword arguments given to the
                                 evaluation function. Default is {} indicating
                                 that no arguments are given.
        trotter_level          : The Trotter level to be used during the time
                                 evolution. Possible values are 1, 2 and 4.
                                 Default is 1.
        abort_at_max_D         : Whether the tMPS calculation should be
                                 finished prematurely after reaching the given
                                 local bond dimension. If set to False, the
                                 calculation will continue but the local bond
                                 dimension will overrule the given acceptable
                                 truncation error. Default is False.
        print_info             : Whether to print the status of the tMPS
                                 calculation to the standard output. Possible
                                 choices are False, '1' and '2' with the
                                 numeric values indicating the level of
                                 verbosity and False indicating that no
                                 information should be printed. Default is
                                 False.
        '''

        self.MPS.set_local_bond_dimension(int(1e6)) # as infinity

        D_here = None if abort_at_max_D else D

        Delta_t = Delta_t if Delta_t is not None else self.Delta_t

        trunc_err, D_max, evaluation_results = [], [], []

        if print_info:
            if print_info != 2:
                print('step'.rjust(5)+' | '+'cum. trunc. err.'.rjust(20)
                      +' | '+'max D'.rjust(5))
                print(36*'-')

        for i in range(time_steps):

            # determine MPOs
            if self._time_dependent:
                MPO = self._create_time_evolution_MPOs_2site(
                    self._f(self._t,*self._f_args,**self._f_kwargs),
                    Delta_t,trotter_level=trotter_level)

            else:

                if (trotter_level != self.trotter_level or
                    Delta_t != self.Delta_t):

                    self._time_evolution_MPOs = (
                        self._create_time_evolution_MPOs_2site(
                            self._f,Delta_t,trotter_level))

                MPO = self._time_evolution_MPOs

            # apply the MPOs
            tr_err = 0
            if self.trotter_level == 1:

                # first order trotterization
                for j in [0,1]:
                    self.MPS.apply_MPO_to_MPS(MPO[j])

                    # compress only affected bonds
                    tr_err += self._compress_MPS(j,D_here,max_truncation_error)

            elif self.trotter_level == 2 and self._time_dependent:

                for j in [0,1,0]:
                    self.MPS.apply_MPO_to_MPS(MPO[j])

                    # compress only affected bonds
                    tr_err += self._compress_MPS(j,D_here,max_truncation_error)

            elif self.trotter_level == 2 and not self._time_dependent:

                # second order trotterization

                # in the time independent case can save 33.33% of time by never
                # explicitly completing a time step until we have to
                # -> e/2 o e/2 e/2 o e/2 -> e/2 o e o e/2.

                if i==0 or (evaluation_step !=-1 and i % evaluation_step == 0):
                    # MPS is at defined time. Use half time in the beginning
                    self.MPS.apply_MPO_to_MPS(MPO[0])
                else:
                    self.MPS.apply_MPO_to_MPS(MPO[1])

                # compress only affected bonds
                tr_err += self._compress_MPS(0,D_here,max_truncation_error)   

                self.MPS.apply_MPO_to_MPS(MPO[2])

                # compress only affected bonds
                tr_err += self._compress_MPS(1,D_here,max_truncation_error)   

                if (i+1) == time_steps or (evaluation_step != -1 and
                                           (i+1) % evaluation_step == 0):

                    # MPS must be at defined time after this.
                    # Use half time at end
                    self.MPS.apply_MPO_to_MPS(MPO[0])

                    # compress only affected bonds
                    tr_err += self._compress_MPS(0,D_here,max_truncation_error)

            elif self.trotter_level == 4:

                # fourth order trotterization

                # ignore time independent case here as it will only save 10%.

                for j in [0,3,1,3,2,4,2,3,1,3,0]:
                    self.MPS.apply_MPO_to_MPS(MPO[j])

                    # compress only affected bonds
                    tr_err += self._compress_MPS(
                        0 if j in [0,1,2] else 1,D_here,max_truncation_error)


            # time step is complete, perform post-processing

            D_max_here = int(max(self.MPS.get_local_bond_dimensions()))

            # store truncation error and maximal local bond dimension
            if evaluation_step != -1 and (i+1) % evaluation_step == 0:
                trunc_err.append(tr_err)
                D_max.append(D_max_here)

            # calculate observables
            if (evaluation_function is not None and evaluation_step != -1
                and (i+1) % evaluation_step == 0):

                evaluation_results.append(evaluation_function(
                    self.MPS,*evaluation_args_list,**evaluation_kwargs_dict))

            if (print_info and evaluation_step != -1 and
                evaluation_step != -1 and (i+1) % evaluation_step == 0):

                if print_info == 2:
                    print('\revolving: {}/{}'.format(i+1,time_steps),end='')
                else:
                    print(str(i+1).rjust(5)+' | '+str(trunc_err[-1]).rjust(20)+
                          ' | '+str(D_max[-1]).rjust(5))

            # prepare MPS for next run
            self.MPS.make_MPS_left_canonical()
            self._t += Delta_t

            if abort_at_max_D and D is not None and D_max_here > D:
                self.MPS.set_local_bond_dimension(D_max_here) # as infinity
                break

        # return results
        if evaluation_function is None:
            return trunc_err, D_max
        else:
            return trunc_err, D_max, evaluation_results
