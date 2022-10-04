import numpy as np
import scipy as sp
import itertools
import copy
import time
import scipy.sparse.linalg as linalg

####################################################################
# save and load reducedTensors to specific positions in HDF5 files #
####################################################################

def save_reducedTensor_hdf5(hdf5_handler,reducedTensor,name):
    '''
    Saves a given reducedTensor object in a HDF5 file to a given position.

    hdf5_handler:  The handler to the HDF5 file (from h5py). It can point
                   to a group within the file.
    reducedTensor: The reducedTensor object to be saved. All sectors and all
                   metadata (such as size or dimensionality) will be saved.
    name:          The name under which the reducedTensor will be saved.
                   A new group under this name will be created. In this group
                   the tensor will be saved. This group has to be referenced
                   in the loading function to load the reducedTensor again.
    '''

    hdf5_handler.create_group(name)
    f = hdf5_handler[name]

    #save attributes
    f.attrs['Q'] = reducedTensor.Q
    f.attrs['ndim'] = reducedTensor.ndim
    f.attrs['size'] = np.prod(reducedTensor.shape)
    f.attrs['shape'] = reducedTensor.shape
    f.attrs['q_signs'] = reducedTensor.q_signs
    f.attrs['filling'] = reducedTensor.filling
    f.attrs['density'] = reducedTensor.density
    f.attrs['num_sectors'] = len(reducedTensor.sectors.keys())
    f.attrs['dtype_str'] = str(reducedTensor.dtype)

    #save charge vectors
    for i in range(reducedTensor.ndim):
        f['q_vector{}'.format(i)] = reducedTensor.q_vectors[i]

    for key,sector in reducedTensor.sectors.items():
        f['sector{}'.format(key)] = sector

def load_reducedTensor_hdf5(hdf5_handler):
    '''
    Load a reducedTensor from a given hdf5_handler (from h5py). The handler
    must point to the group under whose name the reducedTensor was originally
    saved.
    '''

    Q = hdf5_handler.attrs['Q']
    q_signs = list(hdf5_handler.attrs['q_signs'])

    q_vectors = [hdf5_handler['q_vector{}'.format(i)]
                 for i in range(hdf5_handler.attrs['ndim'])]

    q_sectors = {tuple(map(int,key.strip('sector() ').split(','))) :
                 np.array(value) for key,value in hdf5_handler.items()
                 if key.startswith('sector')}

    return reducedTensor(tensor=q_sectors,list_of_q=q_vectors,
                         list_of_xi=q_signs,Q=Q,sectors_given=True)

#################################################
# auxiliary functions regarding (nested) tuples #
#################################################

def mult_tuple(tup,factor):
    '''
    Multiplies all entries of a tuple, a list or scalar with the given factor.
    If a scalar is given, a scalar will be returned. If a tuple or list is
    given a tuple will be returned. The multiplication works iterativerly. If a
    tuple contains tuples, the entries of the nested tuple will be multiplied
    by the given factor as well.
    '''

    if type(tup) is not tuple and type(tup) is not list:
        return tup*factor

    tup_as_list = list(tup)
    for i in range(len(tup_as_list)):

        if type(tup_as_list[i]) is tuple:
            tup_as_list[i] = mult_tuple(tup_as_list[i],factor)
        else:
            tup_as_list[i] *= factor

    return tuple(tup_as_list)

def sum_tuple(tup):
    '''
    Sums all entries of a tuple. If the tuple contains tuples as entries, then
    the elements of the nested tuples will be summed as well.
    '''

    if type(tup) is not tuple and type(tup) is not list:
        return tup

    sum_tup = 0

    for i in tup:
        if type(i) is tuple:
            sum_tup += sum_tuple(i)
        else:
            sum_tup += i
    return sum_tup

def lift_nesting(tup,pos):
    '''
    Lifts the nesting of a tuple at a given position where a nested tuple
    occurs. The internal structure remains unchanged. Further nested tuples are
    preserved. A tuple is returned.
    '''

    return tuple([*tup[:pos],*tup[pos],*tup[pos+1:]])

def purify_tuple(tup):
    '''
    Replace all nested tuples by the sum of the tuple's constituents.
    '''

    list_tup = list(tup)

    for i in range(len(list_tup)):
        if type(list_tup[i]) is tuple:
            list_tup[i] = sum_tuple(list_tup[i])

    return tuple(list_tup)

#######################################################################
# functions that provide additional information about reduced tensors #
#######################################################################

def get_sectors_to_be_joined_if_purified(tensor):
    '''
    Returns a dictionary with pure sector names (no nested tuples) as keys,
    where the value assigned to each key is a list containing all sector names
    of sectors which need to be joined if the reducedTensor is purified, i.e.
    if all nested tuples are resolved. Example: sectors ((-2,2),0) and
    ((2,-2),0) are both named (0,0) if the nesting is resolved. The returned
    dictionary will then have a key (0,0) with value [((-2,2),0) , ((2,-2),0)].
    '''

    # saves the unpurified names of puriefied sectors
    sectors_to_be_joined_dict = {} 

    for sector in tensor.sectors.keys():

        pure_sector = purify_tuple(sector)

        if pure_sector not in sectors_to_be_joined_dict:
            sectors_to_be_joined_dict[pure_sector] = [sector]
        else:
            sectors_to_be_joined_dict[pure_sector].append(sector)

    return sectors_to_be_joined_dict


def get_sub_pipe_dict(tensor,sub_pipeID):
    '''
    Returns a pipe dictionary for all nested subsectors for the given
    pipeID 'sub_pipeID'. The returned dictionary contains for each subsector
    the charge vectors, charge signs as well as the shape.
    '''

    pipe_list = copy.copy(sub_pipeID)
    pipe_sub_dict = {}

    while True:

        if pipe_list == []:
            return pipe_sub_dict

        key = pipe_list.pop(0)
        if key is None:
            continue

        p = tensor.pipe_dict[(key,'p')]
        for p_i in p:
            if p_i is not None:
                pipe_list.append(p_i)

        pipe_sub_dict[(key,'p')] = tensor.pipe_dict[(key,'p')]
        pipe_sub_dict[(key,'q')] = tensor.pipe_dict[(key,'q')]
        pipe_sub_dict[(key,'s')] = tensor.pipe_dict[(key,'s')]

##############################################################
# functions that manipulate instances of class reducedTensor #
##############################################################

def combine_axes_for_reduced_tensor(tensor,combined_sign,start_axis,
                                    number_of_axes_to_the_right):
    '''
    Binds the legs of a tensor together into a pipe. The old charge vectors and
    charge signs will be returned and also stored in the tensor's pipe
    dictionary so that later on the new tensor axis can be split again into the
    axes it was created from. The axes-to-be-bound must lie next to each other.
    The sectors in the tensor are only reshaped but not combined therefore
    becoming the subsectors of the tensor.

    The argument 'start_axis' is the index of the leftmost axis while the
    argument 'number_of_axes_to_the_right' describes the number of axes to be
    combined. Furthermore, the new axis needs a new charge sign, which is
    provided by the argument 'combined_sign'.

    Iterative binding (binding two legs to a pipe and this pipe with another
    leg or pipe to a new pipe) is possible and the iterative structure will be
    preserved in the tensor's pipe dictionary.
    '''

    # update sectors into subsectors (change name and reshape but do not join)
    keys = list(tensor.sectors.keys())
    for sector in keys:

        # determine new shape and new name
        shape = list(np.shape(tensor.sectors[sector]))
        name  = list(sector)
        new_axis_name = []
        new_axis_size = 1

        for i in range(number_of_axes_to_the_right):
            new_axis_size *= shape.pop(start_axis)
            new_axis_name.append(mult_tuple(
              name.pop(start_axis),tensor.q_signs[start_axis+i]*combined_sign))

        shape.insert(start_axis,new_axis_size)
        name.insert(start_axis,tuple(new_axis_name))

        # reshape sector and enter entry in new dictionary
        tensor.sectors[tuple(name)] = tensor.sectors.pop(sector).reshape(shape)


    # calculate new combined charge vector
    aux = [mult_tuple(tensor.q_vectors[start_axis+i],
                      tensor.q_signs[start_axis+i])
           for i in range(number_of_axes_to_the_right)]

    combined_q = [i for i in itertools.product(*aux)]


    # delete old data and store for return
    old_q_vectors = []
    old_q_signs   = []
    for i in range(number_of_axes_to_the_right):
        old_q_vectors.append(tensor.q_vectors.pop(start_axis))
        old_q_signs.append(tensor.q_signs.pop(start_axis))


    # update charge vectors and charge signs
    tensor.q_vectors.insert(start_axis,mult_tuple(combined_q,combined_sign))
    tensor.q_signs.insert(start_axis,combined_sign)


    # update auxiliary information
    tensor.ndim  = len(tensor.q_vectors)
    tensor.shape = tuple([len(q) for q in tensor.q_vectors])

    # update information about pipes
    tensor.pipehighestID += 1
    new_pipe_list = tensor.pipeID[
        start_axis:start_axis+number_of_axes_to_the_right]

    del tensor.pipeID[start_axis:start_axis+number_of_axes_to_the_right]
    tensor.pipeID.insert(start_axis,tensor.pipehighestID)

    tensor.pipe_dict[(tensor.pipehighestID,'p')] = new_pipe_list
    tensor.pipe_dict[(tensor.pipehighestID,'q')] = old_q_vectors
    tensor.pipe_dict[(tensor.pipehighestID,'s')] = old_q_signs

    # return information about old axes
    return old_q_vectors, old_q_signs

def split_axis_for_reduced_tensor(tensor,axis,
                                  list_of_q_vectors=None,list_of_q_signs=None):
    '''
    Split the pipe-axis at position 'axis' into its individual legs. To do
    this, the original list of charge vectors and charge sectors for the axis
    involved is needed. This information may either be provided directly
    through the arguments 'list_of_q_vectors' and 'list_of_q_signs' or, if the
    arguments are set to None is loaded from the pipe dictionary of the tensor.
    
    TODO: By setting 'list_of_q_vectors' and 'list_of_q_signs' accordingly,
    it is possible to split the pipe not into the original axes it was composed
    of but into new and different axes.
    '''

    # Update pipe information and get information about legs
    # if not yet received.
    pipeID = tensor.pipeID[axis]

    if list_of_q_vectors is None:
        list_of_q_vectors = tensor.pipe_dict[(pipeID,'q')]

    if list_of_q_signs is None:
        list_of_q_signs   = tensor.pipe_dict[(pipeID,'s')]

    del tensor.pipeID[axis]
    tensor.pipeID[axis:axis] = tensor.pipe_dict[(pipeID,'p')]

    del tensor.pipe_dict[(pipeID,'p')]

    sign_of_axis = tensor.q_signs[axis]

    # update sectors
    keys = list(tensor.sectors.keys())
    for sector in keys:

        # determine new shape
        shape = list(np.shape(tensor.sectors[sector]))
        del shape[axis]

        shape[axis:axis] = [list_of_q_vectors[i].count(
            mult_tuple(sector[axis][i],list_of_q_signs[i]*sign_of_axis))
                            for i in range(len(list_of_q_vectors))]

        # reshape sector
        tensor.sectors[sector] = tensor.sectors[sector].reshape(shape)

        # rename sector
        new_sector = list(sector)
        new_sector[axis] = [new_sector[axis][i]*list_of_q_signs[i]
                            for i in range(len(list_of_q_signs))]
        new_sector[axis] = mult_tuple(new_sector[axis],tensor.q_signs[axis])

        tensor.sectors[lift_nesting(new_sector,axis)] = (
            tensor.sectors.pop(sector))

    # update charge vectors and signs
    del tensor.q_vectors[axis]
    del tensor.q_signs[axis]
    for i in range(len(list_of_q_vectors)):

        tensor.q_vectors.insert(axis,list_of_q_vectors[-1-i])
        tensor.q_signs.insert(axis,list_of_q_signs[-1-i])

    # update auxiliary information
    tensor.ndim  = len(tensor.q_vectors)
    tensor.shape = tuple([len(q) for q in tensor.q_vectors])

def swapaxes_for_reduced_tensor_inplace(tensor,axis1,axis2):
    '''
    Takes a reducedTensor and swaps the supplied axes. As the array is changed
    inplace, the old reducedTensor will be no longer available after this
    operation.
    '''

    # don't do anything if axes are equal
    if axis1 == axis2:
        return

    # modify charge vectors
    tensor.q_vectors[axis1],tensor.q_vectors[axis2] = (
        list(tensor.q_vectors[axis2]),list(tensor.q_vectors[axis1]))

    # modify charge signs
    tensor.q_signs[axis1],tensor.q_signs[axis2] = (
        tensor.q_signs[axis2],tensor.q_signs[axis1])

    # Swap axes of all sectors.
    # Here we have to be careful to not overwrite any existing adresses.
    keys = tuple(tensor.sectors.keys())
    already_handled = []

    for key in keys:

        if key in already_handled:
            continue

        new_key = list(key)
        new_key[axis1],new_key[axis2] = new_key[axis2],new_key[axis1]
        new_key = tuple(new_key)

        if new_key in keys and not new_key == key:
            tensor.sectors[new_key],tensor.sectors[key] = (
                tensor.sectors[key].swapaxes(axis1,axis2),
                tensor.sectors[new_key].swapaxes(axis1,axis2))
            already_handled.append(new_key)

        else:
            tensor.sectors[new_key] = tensor.sectors.pop(key).swapaxes(axis1,
                                                                       axis2)

    # modify tensor pipe IDs
    tensor.pipeID[axis1],tensor.pipeID[axis2] = (tensor.pipeID[axis2],
                                                 tensor.pipeID[axis1])

    # modify metadata
    s = list(tensor.shape)
    s[axis1],s[axis2] = s[axis2],s[axis1]
    tensor.shape = tuple(s)

def swapaxes_for_reduced_tensor(tensor,axis1,axis2):
    '''
    Takes a reducedTensor and swaps the supplied axis. The sector dictionary is
    here build anew and a new reducedTensor is returned.
    '''

    # create new charge vectors
    q_vectors = copy.copy(tensor.q_vectors)
    aux = q_vectors[axis1]
    q_vectors[axis1] = q_vectors[axis2]
    q_vectors[axis2] = aux

    # create new charge signs
    q_signs = copy.copy(tensor.q_signs)
    aux = q_signs[axis1]
    q_signs[axis1] = q_signs[axis2]
    q_signs[axis2] = aux

    # create new sector dict
    new_sectors = {}
    for sector in tensor.sectors.keys():

        # update keys
        aux = list(sector)
        aux2 = aux[axis1]
        aux[axis1] = aux[axis2]
        aux[axis2] = aux2
        new_sectors[tuple(aux)] = tensor.sectors[sector].swapaxes(axis1,axis2)

    # copy old pipe dict and interchange entries of axis1 and axis2
    newPipeID = copy.copy(tensor.pipeID)
    aux = newPipeID[axis1]
    newPipeID[axis1] = newPipeID[axis2]
    newPipeID[axis2] = aux

    return reducedTensor(tensor=new_sectors,list_of_q=q_vectors,
                         list_of_xi=q_signs,Q=tensor.Q,sectors_given=True,
                         pipeID=newPipeID,
                         pipehighestID=tensor.pipehighestID,
                         pipe_dict=tensor.pipe_dict)

def contract_reduced_tensors_along_axes(tensor1,tensor2,axis1_list,axis2_list,
                                        capture_garden_eden=True,
                                        for_DMRG=False):
    '''
    Sums the two given tensors along the supplied axes. Multiple axes at once
    can be used. Only valid fillable sectors will be stored. Nonfillable valid
    sectors (valid sectors which contain only zeros) will only be identified
    and set to a zero matrix of the appropriate size if 'capture_garden_eden'
    is set to True. If not, these sectors will be missing. Omiting the capture
    of such sectors makes the contraction faster and returns a smaller
    reducedTensor object.

    For the slow running DMRG (fourth power scaling) the last contraction done
    to receive the effective Hamiltonian, can be made faster by omiting a lot
    of calculations which are required to perform the contraction accurately
    but are not required to perform the DMRG accurately. In this sense, time
    can be saved by foregoing unneeded numerical work. Setting for_DMRG=False,
    calculates the contraction normally. To make use of the above-explained
    feature, set for_DMRG=1 for one-site DMRG and for_DMRG=2 for two-site DMRG.
    '''

    # perform consistency checks
    if len(axis1_list) != len(axis2_list):
        raise ValueError('axis1_list and axis2_list must both have the same '
                         'length but have lengths {}'.format(len(axis1_list))+
                         ' and {}.'.format(len(axis2_list)))

    if len(axis1_list) == 0:
        raise ValueError('The axis lists must contain at least one item.')

    for i in range(len(axis1_list)):
        # The axes over which we sum need to have the same charge vectors but
        # opposite signs.
        if (list(tensor1.q_vectors[axis1_list[i]]) !=
            list(tensor2.q_vectors[axis2_list[i]])):
            raise ValueError('The charge vector for axis '
                             '{} of tensor1 must be the'.format(axis1_list[i])+
                             ' same as for axis {} of '.format(axis2_list[i])+
                             'tensor2, but is not.')

        # The charge signs need to be different for the axes-to-be-summed-over
        if tensor1.q_signs[axis1_list[i]] == tensor2.q_signs[axis2_list[i]]:
            raise ValueError('The charge sign for axis '
                             '{} of tensor1 must be '.format(axis1_list[i])+
                             'different to the charge sign for axis '
                             '{} of tensor2, but is '.format(axis2_list[i])+
                             'not.')


    # create infrastructure for new tensor
    new_q_signs   = []
    new_q_vectors = []
    for i in range(len(tensor1.q_signs)):
        if i not in axis1_list:
            new_q_signs.append(tensor1.q_signs[i])
            new_q_vectors.append(tensor1.q_vectors[i])

    for i in range(len(tensor2.q_signs)):
        if i not in axis2_list:
            new_q_signs.append(tensor2.q_signs[i])
            new_q_vectors.append(tensor2.q_vectors[i])

    new_Q = tensor1.Q + tensor2.Q

    # to find new valid sectors, stop making axes we sum over contribute
    signs1 = np.copy(tensor1.q_signs)
    signs2 = np.copy(tensor2.q_signs)

    for index in axis1_list:
        signs1[index] = 0

    for index in axis2_list:
        signs2[index] = 0


    # perform the summation
    sectors = {}

    # find all sectors with parents
    for s1 in tensor1.list_sectors():

        sum1 = np.dot(purify_tuple(s1),signs1)

        axis1_entries = [s1[i] for i in axis1_list]

        for s2 in tensor2.list_sectors():

            axis2_entries = [s2[i] for i in axis2_list]

            # if the entries of the contracted axes don't match, abbort
            if axis1_entries != axis2_entries:
                continue

            sum2 = np.dot(purify_tuple(s2),signs2)

            # check if illegal sector (can actually never happen)
            if sum1 + sum2 != tensor1.Q + tensor2.Q:
                continue

            list1 = list(s1)
            for index in sorted(axis1_list, reverse=True):
                del list1[index]

            list2 = list(s2)
            for index in sorted(axis2_list, reverse=True):
                del list2[index]

            sector_name = tuple(list1+list2)

            # ignore if not part of the (0,0) pure sector
            # if shaped into quadratic matrix
            if for_DMRG:
                if (for_DMRG == 2 and
                    new_q_signs[2]*sector_name[2] +
                    new_q_signs[4]*sector_name[4] +
                    new_q_signs[1]*sector_name[1] +
                    new_q_signs[7]*sector_name[7] != 0):
                    continue

                elif (for_DMRG == 1 and
                      new_q_signs[2]*sector_name[2] +
                      new_q_signs[1]*sector_name[1] +
                      new_q_signs[5]*sector_name[5] != 0):
                    continue

            if sector_name in sectors:
                sectors[sector_name] += np.real_if_close(np.tensordot(
                    tensor1.sectors[s1],tensor2.sectors[s2],
                    axes=(axis1_list,axis2_list)))
            else:
                sectors[sector_name] = np.tensordot(tensor1.sectors[s1],
                                                    tensor2.sectors[s2],
                                                    axes=(axis1_list,
                                                          axis2_list))


    # find all sectors without parents
    if capture_garden_eden:

        # create the unique q_{i}
        list_of_s = []
        num_sectors = 1
        for q in new_q_vectors:
            list_of_s.append(list(set(q)))
            num_sectors *= len(list_of_s[-1])

        # iterate through all sectors
        # for dim_set in range(num_sectors):

        # Calculate the set of the purified charge vectors and multiply it with
        # the signs. This reduces the amount of sectors we have to test (as
        # duplicates are removed) and each check is faster as we do not have to
        # perform the multiplication with the signs so often. Purifing the
        # charge vectors reduces the amount of checks and time spend for each
        # check as well. This reduces the loop to a minimum of iterations.

        # If the sum is equal to the total charge, the sector is valid. We have
        # to remove the signs from sectorname again by multiplying the signs a
        # second time. We also have to 'unpurify' the sector name again. To do
        # this we need to save the signs for the new tensor explicitly and also
        # store in a dictionary all pipe configurations for each leg
        # configuration.

        ndim = tensor1.ndim+tensor2.ndim-len(axis1_list)-len(axis2_list)

        # charge vectors of new tensor
        q_vec =      [vec for i,vec in enumerate(tensor1.q_vectors)
                      if i not in axis1_list]

        q_vec.extend([vec for i,vec in enumerate(tensor2.q_vectors)
                      if i not in axis2_list])

        # get multiplicity dictionary-dictionary
        mult = {}
        for n in range(ndim):

            q_vec_set = list(set(q_vec[n]))
            q_vec_set_pure = purify_tuple(q_vec_set)

            for index,entry in enumerate(q_vec_set):

                a = (n,q_vec_set_pure[index])
                b = entry
                c = q_vec[n].count(entry)
                if a in mult:
                    mult[a][b] = c
                else:
                    mult[a] = {b:c}

        # get the charge sign-vector sets
        q_vec_set =      [set(tensor1.q_signs[i]*np.array(purify_tuple(vec)))
                          for i,vec in enumerate(tensor1.q_vectors)
                          if i not in axis1_list]

        q_vec_set.extend([set(tensor2.q_signs[i]*np.array(purify_tuple(vec)))
                          for i,vec in enumerate(tensor2.q_vectors)
                          if i not in axis2_list])

        # get the signs
        q_signs =      [s for i,s in enumerate(tensor1.q_signs)
                        if i not in axis1_list]

        q_signs.extend([s for i,s in enumerate(tensor2.q_signs)
                        if i not in axis2_list])

        q_signs = np.array(q_signs)

        # get all sectors
        Q = tensor1.Q + tensor2.Q
        for sector_attempt in itertools.product(*q_vec_set):

            if sum(sector_attempt) == Q:
                sector = q_signs*sector_attempt

                # find out names and sizes of subsectors
                subsector_keys = [mult[(n,entry)].keys()
                                  for n,entry in enumerate(sector)]

                for subsector in itertools.product(*subsector_keys):

                    if subsector not in sectors:

                        size = [mult[(index,sector[index])][subsector[index]]
                                for index in range(len(subsector))]

                        sectors[subsector] = np.zeros(size)

    # Create pipe infrastructure for new tensor
    pipeID1 = copy.copy(tensor1.pipeID)
    pipeID2 = copy.copy(tensor2.pipeID)

    for i in np.sort(axis1_list)[::-1]:
        del pipeID1[i]

    for i in np.sort(axis2_list)[::-1]:
        del pipeID2[i]

    newPipeID = pipeID1.extend(pipeID2)
    newPipehighestID = tensor1.pipehighestID + tensor2.pipehighestID

    newPipe_dict = copy.copy(tensor1.pipe_dict)

    for i in tensor2.pipe_dict.keys():
        new_key = list(i)
        new_key[0] += tensor1.pipehighestID
        newPipe_dict[tuple(new_key)] = copy.copy(tensor2.pipe_dict[i])

    # create new reducedTensor object and return it
    new_tensor = reducedTensor(tensor=sectors,list_of_q=new_q_vectors,
                               list_of_xi=new_q_signs,Q=new_Q,
                               sectors_given=True,pipeID=newPipeID,
                               pipehighestID=newPipehighestID,
                               pipe_dict=newPipe_dict,dtype='ref')

    return new_tensor

def SVD_for_reduced_matrix(matrix, D=None, max_truncation_error=None,
                           normalize=None):
    '''
    Calculate the singular value decomposition (SVD) of a reduced matrix. By
    definition, the charge of U will be zero, while the charge of V will be the
    same as for the supplied matrix. The charge vector of the newly created
    axis will be the same as for the first axis, albeit in purified form. The
    sign which accomodates the new axis is the negative of the sign of the
    first axis. There are several possibilities to post-process the results of
    the SVD. First, the size of the newly introduced dimension can be reduced
    by D (the new size) and max_truncation_error (the highest acceptable
    error). If both are set, D is the stronger parameter meaning that the new
    dimension will not be larger than D, even if this means violating the
    condition that the truncation error should not be larger than
    max_truncation_error. Another possibility to post-process the results is by
    multiplying the singular values into V (normalisation='left', 'l') or
    multiplying them into U ('right', 'r'). The square root of the singular
    values can also be multiplied to both U and V ('none', 'n') or not at all
    ('both','b', None).

    Returns U,S and V, where U and V are reduced matrices, while S is the list
    of singular values. S will be returned regardless if it got multiplied
    somewhere. The resulting truncation error will also be returned which is
    0.0 if no singular values are removed.
    '''

    ## initialise storage variables
    U_pure_dict = {}
    V_pure_dict = {}

    Sdict = {} # stores singular values for each sector.

    Svalue_pure = [] # stores singular values
    Sname_pure  = [] # stores the pure sector where the singular value of same
    #                # position in Svalue_pure was found

    # singular values are stored two-fold. This seems wasting and it is but it
    # makes things later on easier to understand and implement.

    q0name = {} # the first part of the name of the subsectors
    q0size = {} # the size in direction 0 for the subsectors
    q1name = {} # the second part of the name of the subsectors
    q1size = {} # the size in direction 1 for the subsectors


    # perform SVDs
    purified_sector_dict = get_sectors_to_be_joined_if_purified(matrix)
    for pure_sector_name in purified_sector_dict.keys():


        # find out sizes and names of subsectors
        q0name[pure_sector_name] = list(set([item[0] for item in
                                             purified_sector_dict[
                                                 pure_sector_name]]))

        q1name[pure_sector_name] = list(set([item[1] for item in
                                             purified_sector_dict[
                                                 pure_sector_name]]))

        q0size[pure_sector_name] = [np.shape(matrix.sectors[
            tuple([i,q1name[pure_sector_name][0]])])[0]
                                    for i in q0name[pure_sector_name]]

        q1size[pure_sector_name] = [np.shape(matrix.sectors[
            tuple([q0name[pure_sector_name][0],i])])[1]
                                    for i in q1name[pure_sector_name]]


        # create pure sector from subsectors
        pure_sector = np.empty([sum(q0size[pure_sector_name]),
                                sum(q1size[pure_sector_name])],
                               dtype=matrix.dtype)

        filling_dim0 = 0
        for subsector0 in q0name[pure_sector_name]:
            filling_dim1 = 0
            for subsector1 in q1name[pure_sector_name]:

                sector = tuple([subsector0,subsector1])
                sector_shape = matrix.sectors[sector].shape
                pure_sector[filling_dim0:filling_dim0+sector_shape[0],
                            filling_dim1:filling_dim1+sector_shape[1]] = (
                                matrix.sectors[sector])

                filling_dim1 += sector_shape[1]
            filling_dim0 += sector_shape[0]

        # perform SVD on pure sector and store all the results

        # It can happen that the SVD does not converge even if all entries are
        # valid, i.e. no NaN or +-infty are present. In this case, modify the
        # matrix very slightly and feet the modified matrix back into the SVD.
        # That may work. The change is small enough, so that the result remains
        # essentially the same. Deviations from the 'true result' will most
        # likely be consumed by future numerical rounding errors.

        try:
            U,S,V = np.linalg.svd(pure_sector, full_matrices=False,
                                  compute_uv=True)
        except np.linalg.LinAlgError:
            print('SVD did not converge. Trying again with matrix rounded to '
                  '15 decimal numbers. That may help.')
            try:
                U,S,V = np.linalg.svd(np.round(pure_sector,15),
                                      full_matrices=False, compute_uv=True)
            except np.linalg.LinAlgError:
                for i in range(10):
                    print('SVD failded again. Try again with random noise. '
                          'Try {}/10.'.format(i+1))
                    try:
                        U,S,V = np.linalg.svd(pure_sector
                         + (np.random.random(np.shape(pure_sector))-0.5)*1e-15,
                                              full_matrices=False,
                                              compute_uv=True)
                        break
                    except np.linalg.LinAlgError as e:
                        # At one point we just have to abbort
                        if i == 9:
                            raise e

        # store results of the SVD
        U_pure_dict[pure_sector_name] = U
        V_pure_dict[pure_sector_name] = V

        Svalue_pure.extend(S)
        Sname_pure.extend(len(S)*[pure_sector_name])
        Sdict[pure_sector_name] = S


    # Sort the singular values lowest to highest
    S_order = np.argsort(Svalue_pure)

    Svalue_pure = np.array([Svalue_pure[i] for i in S_order])
    Sname_pure  = [Sname_pure[i] for i in S_order]


    # Find out how many singular values have to go and where
    cut_S = 0 if D is None else len(Svalue_pure) - D
    error = 0
    delete_dict = {} # store how many singular values need to be deleted
    #                # for each pure sector

    norm_old = np.sum(Svalue_pure**2)
    if max_truncation_error is not None:

        for i,s in enumerate(Svalue_pure):

            if error + s**2/norm_old > max_truncation_error:
                cut_S = max(cut_S,i)
                break
            else:
                error += s**2/norm_old
                continue

    for i in range(cut_S):
        if Sname_pure[i] in delete_dict.keys():
            delete_dict[Sname_pure[i]] += 1
        else:
            delete_dict[Sname_pure[i]] = 1


    # compress the tensor by cutting cut_S smallest singular values away
    Svalue_pure = Svalue_pure[::-1]
    Sname_pure  = Sname_pure[::-1]

    if cut_S > 0:
        norm_new         = np.sum(Svalue_pure[:-cut_S]**2)
        truncation_error = np.sum(Svalue_pure[-cut_S:]**2)/norm_old
        Sname_pure       = Sname_pure[:-cut_S]
        Svalue_pure      = (Svalue_pure[:-cut_S]/np.sqrt(norm_new)
                            *np.sqrt(norm_old))
    else:
        truncation_error = 0.0
        norm_new         = norm_old

    for s,n in delete_dict.items():
        if n == np.shape(U_pure_dict[s])[1]:
            # delete sector
            del U_pure_dict[s]
            del V_pure_dict[s]
        else:
            # cut sector
            U_pure_dict[s] = U_pure_dict[s][:,:-n]
            V_pure_dict[s] = V_pure_dict[s][:-n]


    # split U pure sectors into U subsectors and apply normalisation behavior
    U_sub_dict = {}
    for pure_sector in U_pure_dict.keys():

        # multiply S into U if wanted (S needs to be cut for this)
        if (normalize == 'right') or (normalize == 'r'):

            S = (Sdict[pure_sector][:np.shape(U_pure_dict[pure_sector])[1]]*
                 np.sqrt(norm_old)/np.sqrt(norm_new))

            U_pure_dict[pure_sector] = np.dot(U_pure_dict[pure_sector],
                                              np.diag(S))

        elif (normalize == 'none') or (normalize == 'n'):

            S = (Sdict[pure_sector][:np.shape(U_pure_dict[pure_sector])[1]]*
                 np.sqrt(norm_old)/np.sqrt(norm_new))

            U_pure_dict[pure_sector] = np.dot(U_pure_dict[pure_sector],
                                              np.diag(np.sqrt(S)))

        start_pos = 0
        for i in range(len(q0name[pure_sector])):

            new_name = tuple([q0name[pure_sector][i],pure_sector[0]])
            size = q0size[pure_sector][i]

            U_sub_dict[new_name] = (
                U_pure_dict[pure_sector][start_pos:start_pos+size])

            start_pos += size


    ## split V pure sectors into V subsectors and apply normalisation behavior
    V_sub_dict = {}
    for pure_sector in V_pure_dict.keys():

        # multiply S into V if wanted (S needs to be cut for this)
        if (normalize == 'left') or (normalize == 'l'):

            S = (Sdict[pure_sector][:np.shape(V_pure_dict[pure_sector])[0]]*
                 np.sqrt(norm_old)/np.sqrt(norm_new))

            V_pure_dict[pure_sector] = np.dot(np.diag(S),
                                              V_pure_dict[pure_sector])

        elif (normalize == 'none') or (normalize == 'n'):

            S = (Sdict[pure_sector][:np.shape(V_pure_dict[pure_sector])[0]]*
                 np.sqrt(norm_old)/np.sqrt(norm_new))

            V_pure_dict[pure_sector] = np.dot(np.diag(np.sqrt(S)),
                                              V_pure_dict[pure_sector])

        start_pos = 0
        for i in range(len(q1name[pure_sector])):

            new_name = tuple([pure_sector[0],q1name[pure_sector][i]])
            size = q1size[pure_sector][i]

            V_sub_dict[new_name] = (
                V_pure_dict[pure_sector][:,start_pos:start_pos+size])

            start_pos += size


    ## initialise reducedTensor objects for U and V
    q_list = matrix.q_vectors
    s_list = matrix.q_signs
    q_new = [entry[0] for entry in Sname_pure]

    # create pipe infrastructure for new tensors

    pipeID_U = [matrix.pipeID[0],None]
    pipeID_V = [None,matrix.pipeID[1]]

    pipe_dict_U = get_sub_pipe_dict(matrix,pipeID_U)
    pipe_dict_V = get_sub_pipe_dict(matrix,pipeID_V)

    U = reducedTensor(tensor=U_sub_dict,list_of_q=[q_list[0],q_new],
                      list_of_xi=[s_list[0],-s_list[0]],Q=0,sectors_given=True,
                      pipeID=pipeID_U,pipehighestID=matrix.pipehighestID,
                      pipe_dict=pipe_dict_U)

    V = reducedTensor(tensor=V_sub_dict,list_of_q=[q_new,q_list[1]],
                      list_of_xi=s_list,Q=matrix.Q,sectors_given=True,
                      pipeID=pipeID_V,pipehighestID=matrix.pipehighestID,
                      pipe_dict=pipe_dict_V)

    return U,Svalue_pure,V,truncation_error

def QR_for_reduced_matrix(matrix):
    '''
    Calculate the QR decomposition of a reduced matrix. By definition, the
    charge of Q will be zero, while the charge of R will be the same as for the
    supplied matrix. The charge vector of the newly created axis will be the
    same as for the first axis, albeit in purified form. The sign which
    accomodates the new axis is the negative of the sign of the first axis.
    Returns Q and R which are reduced matrices.
    '''

    # initialise storage variables
    U_sub_dict = {}
    V_sub_dict = {}
    q_new = []

    # perform QRs
    purified_sector_dict = get_sectors_to_be_joined_if_purified(matrix)
    for pure_sector_name in purified_sector_dict.keys():

        # find out sizes and names of subsectors
        q0name = list(set([item[0] for item in
                           purified_sector_dict[pure_sector_name]]))

        q1name = list(set([item[1] for item in
                           purified_sector_dict[pure_sector_name]]))

        q0size = [np.shape(matrix.sectors[tuple([i,q1name[0]])])[0]
                  for i in q0name]

        q1size = [np.shape(matrix.sectors[tuple([q0name[0],i])])[1]
                  for i in q1name]


        # create pure sector from subsectors
        pure_sector = np.empty([sum(q0size),sum(q1size)],dtype=matrix.dtype)

        filling_dim0 = 0
        for subsector0 in q0name:
            filling_dim1 = 0
            for subsector1 in q1name:

                sector = tuple([subsector0,subsector1])
                sector_shape = matrix.sectors[sector].shape
                pure_sector[filling_dim0:filling_dim0+sector_shape[0],
                            filling_dim1:filling_dim1+sector_shape[1]] = (
                                matrix.sectors[sector])

                filling_dim1 += sector_shape[1]
            filling_dim0 += sector_shape[0]

        # perform QR on pure sector and store all the results
        U,V = np.linalg.qr(pure_sector) # (U=Q, V=R)

        # split U into subsectors
        start_pos = 0
        for i in range(len(q0name)):

            new_name = tuple([q0name[i],pure_sector_name[0]])
            size = q0size[i]

            U_sub_dict[new_name] = U[start_pos:start_pos+size]
            start_pos += size

        # split V into subsectors
        start_pos = 0
        for i in range(len(q1name)):

            new_name = tuple([pure_sector_name[0],q1name[i]])
            size = q1size[i]

            V_sub_dict[new_name] = V[:,start_pos:start_pos+size]
            start_pos += size

        # create charge vector for new dimension iteratively.
        q_new.extend(np.shape(U)[1]*[pure_sector_name[0]])

    ## initialise reducedTensor objects for U and V
    q_list = matrix.q_vectors
    s_list = matrix.q_signs

    # create pipe infrastructure for new tensors
    pipeID_U = [matrix.pipeID[0],None]
    pipeID_V = [None,matrix.pipeID[1]]

    pipe_dict_U = get_sub_pipe_dict(matrix,pipeID_U)
    pipe_dict_V = get_sub_pipe_dict(matrix,pipeID_V)

    U = reducedTensor(tensor=U_sub_dict,list_of_q=[q_list[0],q_new],
                      list_of_xi=[s_list[0],-s_list[0]],Q=0,sectors_given=True,
                      pipeID=pipeID_U,pipehighestID=matrix.pipehighestID,
                      pipe_dict=pipe_dict_U)

    V = reducedTensor(tensor=V_sub_dict,list_of_q=[q_new,q_list[1]],
                      list_of_xi=s_list,Q=matrix.Q,sectors_given=True,
                      pipeID=pipeID_V,pipehighestID=matrix.pipehighestID,
                      pipe_dict=pipe_dict_V)

    return U,V

def groundstate_searcher_for_reduced_matrix_test(matrix,v0):
    '''
    A simple help function to test the accuracy of the function
    groundstate_searcher_for_reduced_matrix. Takes a reduced matrix and a
    reduced vector, reconstructs the matrix in dense form and performs an exact
    diagonalisation. The ground state is then brought inte reduced form, using
    the same charge vectors and charge signs as the reduced vector v0.

    Returns the ground state energy and the ground state as reduced tensor.
    '''

    w,v   = np.linalg.eigh(matrix.reconstruct_full_tensor())

    GS  = v[:,0]
    GSE = w[0]

    return GSE, reducedTensor(GS,v0.q_vectors,v0.q_signs,Q=0,pipeID=v0.pipeID,
                              pipehighestID=v0.pipehighestID,
                              pipe_dict=v0.pipe_dict)

def groundstate_searcher_for_reduced_matrix(matrix,v0,method='g',
                                            save_memory=False):
    '''
    Iteratively search for the ground state of 'matrix' which is an instance of
    reducedTensor and must be a square matrix. The argument 'v0', also an
    instance of reducedTensor, serves as the initial guess and must be a vector
    of corresponding length. The argument 'method' describes the exact method
    being used and may take one of the following entries.

    'g'     : Optimise toward ground state (traditional DMRG step).
    'h'     : Optimise toward highest energy eigenstate (same as 'g' for -H).
    'DMRGX' : Used to perform a DMRG-X step. Does not return the ground state
              of H but instead the eigenstate of H closest in terms of spatial
              overlap to v0. Can only by used if H is of type np.ndarray or
              rt.reducedTensor.

    If save_memory=True, this function delets sector of 'matrix' that are no
    longer required in order to free memory as early as possible and as much as
    possible.

    This function returns the tuple (w,v) with w being the energy eigenvalue of
    the newly calculated eigenstate v of H. Here, 'v' will be of the same type
    than 'v0'.
    '''

    ## initialise storage variables
    q0name = {} # the first part of the name of the subsectors
    q0size = {} # the size in direction 0 for the subsectors


    ## search for ground state
    GS   = None # ground state
    GSE  = None # ground state energy
    GSPS = None # pure sector name of the current ground state
    HOV  = None # highest overlap

    purified_sector_dict = get_sectors_to_be_joined_if_purified(matrix)
    for pure_sector_name in purified_sector_dict.keys():

        # Only the (0,0) sector is relevant
        if pure_sector_name[0] != 0:
            continue

        # find out sizes and names of subsectors
        q0name[pure_sector_name] = list(
            set([item[0] for item in purified_sector_dict[pure_sector_name]]))

        q0size[pure_sector_name] = []

        # find out sizes of subsectors
        for i in q0name[pure_sector_name]:
            for j in q0name[pure_sector_name]:
                sector_name = tuple([i,j])
                if sector_name in matrix.sectors:
                    q0size[pure_sector_name].append(
                        np.shape(matrix.sectors[sector_name])[0])
                    break

        # create pure sector from subsectors
        pure_sector    = np.zeros([sum(q0size[pure_sector_name]),
                                   sum(q0size[pure_sector_name])])

        v0_pure_sector = np.zeros([sum(q0size[pure_sector_name])])

        # purify v0 sector
        filling_dim_v0 = 0
        for i,subsector0 in enumerate(q0name[pure_sector_name]):

            sector_v0 = (subsector0,)
            sector_shape_v0 = q0size[pure_sector_name][i]
            if sector_v0 in v0.sectors:
                v0_pure_sector[filling_dim_v0:filling_dim_v0+sector_shape_v0]=(
                    np.real_if_close(v0.sectors[sector_v0]))
            else:
                v0_pure_sector[filling_dim_v0:filling_dim_v0+sector_shape_v0]=(
                    np.zeros(sector_shape_v0))

            filling_dim_v0 += sector_shape_v0

        # build pure sector of matrix
        sector_name = []
        filling_dim0 = 0
        for i,subsector0 in enumerate(q0name[pure_sector_name]):
            filling_dim1 = 0

            for j,subsector1 in enumerate(q0name[pure_sector_name]):

                sector = (subsector0,subsector1)
                sector_shape = (q0size[pure_sector_name][i],
                                q0size[pure_sector_name][j])

                if sector in matrix.sectors:
                    pure_sector[filling_dim0:filling_dim0+sector_shape[0],
                                filling_dim1:filling_dim1+sector_shape[1]] = (
                                    np.real_if_close(matrix.sectors[sector]))

                    # save memory by deleting sectors from matrix
                    if save_memory:
                        del matrix.sectors[sector]

                filling_dim1 += sector_shape[1]
            filling_dim0 += sector_shape[0]
            sector_name.append(subsector0)

        # solve pure sector for given method
        if method in ['g','h']:

            which = 'SA' if method == 'g' else 'LA'

            if np.shape(pure_sector) == (1,1):
                w,v = np.linalg.eigh(pure_sector)
            else:
                w,v = linalg.eigsh(A=pure_sector,v0=v0_pure_sector,k=1,
                                   which=which,tol=0,sigma=None,
                                   maxiter=100*len(pure_sector),ncv=10)

            # If a lower ground state energy is not found, continue search
            if (GSE is not None) and (w[0] > GSE):
                continue

            GS   = copy.deepcopy(v[:,0])
            GSE  = w[0]
            GSPS = pure_sector_name

        elif method == 'DMRGX':

            dimM = np.shape(v0_pure_sector)
            w,v = np.linalg.eigh(pure_sector)

            overlap = np.abs(np.tensordot(np.conjugate(v0_pure_sector),
                                          v,axes=[[0],[0]]))

            highest_overlap_index = np.argmax(overlap)
            highest_overlap = overlap[highest_overlap_index]
            if (HOV is not None) and (highest_overlap < HOV):
                continue

            GS   = copy.deepcopy(v[:,highest_overlap_index])
            GSE  = w[highest_overlap_index]
            HOV  = highest_overlap
            GSPS = pure_sector_name


    # slice ground state back into sub_sectors
    GSS = {}
    start_pos = 0

    for i in range(len(q0name[GSPS])):

        new_name = (q0name[GSPS][i],)
        size = q0size[GSPS][i]

        GSS[new_name] = GS[start_pos:start_pos+size]
        start_pos += size

    pipeID_GSE = [matrix.pipeID[0]]
    pipe_dict_GSE = get_sub_pipe_dict(matrix,pipeID_GSE)

    state = reducedTensor(tensor=GSS,list_of_q=[matrix.q_vectors[0]],
                          list_of_xi=[matrix.q_signs[0]],Q=0,
                          sectors_given=True, pipeID=pipeID_GSE,
                          pipehighestID=matrix.pipehighestID,
                          pipe_dict=pipe_dict_GSE)

    return GSE,state

def trace_for_reduced_matrix(matrix):
    '''
    Calculates and returns the trace of the given argument 'matrix' which must
    be an instance of reducedTensor and represent a square matrix.
    '''

    # perform consistency checks
    if type(matrix) is not reducedTensor:
        raise TypeError('The given matrix must be of type reducedTensor '
                        'but is not')

    if matrix.ndim != 2:
        raise TypeError('The given matrix must be of 2 dimensions but is of '
                        '{} dimensions'.format(matrix.ndim))

    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError('The given matrix must be quadratic but is not.')

    # calculate and return trace
    trace = 0
    for i in range(matrix.shape[0]):
        trace += matrix.get_element_in_tensor_coords([i,i])

    return trace

def isReducedIdentityMatrix(matrix):
    '''
    Determines whether the given reducedTensor 'matrix' is the identity matrix.
    For this, the largest deviation from the identity matrix is determined and
    then returned. The smaller the returned value is, the better does 'matrix'
    approximates the identity matrix and precisely is the identity matrix if
    the returned value is 0.0.
    '''

    size = np.shape(matrix)[0]
    
    missing_diagonal_element = False

    max_error = 0
    
    # try to set diagonal to zero
    for i in range(size):
        max_error = max(max_error,
                        matrix.set_element_in_tensor_coords([i,i]) - 1)
        
        if (matrix.set_element_in_tensor_coords([i,i],0,
                                                dontRaiseError=True) == 1):
            missing_diagonal_element = True

    max_error = max(max_error,1) if missing_diagonal_element else max_error

    # check the largest deviation from one.
    for sector in matrix.sectors.keys():
        error = np.max(np.abs(matrix.sectors[sector]))
        max_error = max(max_error,error)

    return max_error

def matrix_exponential_QUARANTINED(matrix):
    '''
    Check this function: I'm not sure if this really calculates the matrix
    exponential. It calculates the exponential for each sector. Is it possible
    to do it like this?

    Calculate the matrix exponential for a given reducedTensor 'matrix' which
    must be a square matrix. This only works, if the sectors in 'matrix' are
    all pure and no subsectors are being used. Returns the matrix exponential
    of the given matrix as a reducedTensor.
    '''

    if matrix.ndim != 2:
        raise ValueError("'matrix' must have two dimensions.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("'matrix must be quadratic.'")

    if matrix.q_vectors[0] != matrix.q_vectors[1]:
        raise ValueError("'matrix' must be completely symmetric. Both charge "
                         "vectors must be identical.")

        # Not really true but completely different charge vectors would kill
        # the reduced tensor aspect. Could preserve reduced tensor aspect even
        # if q[0] != q[1] for some weird configurations including charge signs
        # and total charge but I'm too lazy right now to properly include them.

    exp_matrix = copy.deepcopy(matrix)

    for key in matrix.sectors.keys():
        exp_matrix.sectors[key] = sp.linalg.expm(exp_matrix.sectors[key])

    return exp_matrix

def multiply_by_scalar(tensor,n):
    '''
    Multiply the given reducedTensor 'tensor' by the given scalar 'n'.
    Returns a new instance of reducedTensor representing the product.
    '''

    tensor2 = copy.deepcopy(tensor)

    # loop through charge sectors and multiply them by n
    for key in tensor2.sectors.keys():
        tensor2.sectors[key] = tensor2.sectors[key] * n

    return tensor2


def insert_dummy_dimension(tensor,pos,sign=1):
    '''
    Insert a dummy dimension into the reducedTensor 'tensor' at position 'pos'
    inplace. The given sign will be used as the sign for this dimension's
    charge vector. It is set to '+1' by default and the charge vector is set
    to '[0]'.
    '''

    # update meta information
    tensor.ndim += 1
    new_shape = list(tensor.shape)
    new_shape.insert(pos,1)
    tensor.shape = tuple(new_shape)
    tensor.pipeID.insert(pos,None)

    # update the charge signs
    tensor.q_signs.insert(pos,sign)

    # insert charge vector
    tensor.q_vectors.insert(pos,[0])

    # update all sectors
    sector_dict_items = [[key,value] for key,value in tensor.sectors.items()]

    for key,value in sector_dict_items:

        new_key = list(key)
        new_key.insert(pos,0)
        new_key = tuple(new_key)

        new_shape = list(np.shape(value))
        new_shape.insert(pos,1)

        tensor.sectors[new_key] = value.reshape(new_shape)
        del tensor.sectors[key]


def show_sectors_of_reduced_matrix(matrix):
    '''
    Prints the location of the sectors of a reduced matrix in the reduced
    matrixon the standard output.
    '''

    shape = matrix.shape

    symbols=['A' , 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N' , 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
             '1A','1B','1C','1D','1E','1F','1G','1H','1I','1J','1K','1L','1M',
             '1N','1O','1P','1Q','1R','1S','1T','1U','1V','1W','1X','1Y','1Z',
             '2A','2B','2C','2D','2E','2F','2G','2H','2I','2J','2K','2L','2M',
             '2N','2O','2P','2Q','2R','2S','2T','2U','2V','2W','2X','2Y','2Z',
             '3A','3B','3C','3D','3E','3F','3G','3H','3I','3J','3K','3L','3M',
             '3N','3O','3P','3Q','3R','3S','3T','3U','3V','3W','3X','3Y','3Z',
             '4A','4B','4C','4D','4E','4F','4G','4H','4I','4J','4K','4L','4M',
             '4N','4O','4P','4Q','4R','4S','4T','4U','4V','4W','4X','4Y','4Z',
             '5A','5B','5C','5D','5E','5F','5G','5H','5I','5J','5K','5L','5M',
             '5N','5O','5P','5Q','5R','5S','5T','5U','5V','5W','5X','5Y','5Z',
             '6A','6B','6C','6D','6E','6F','6G','6H','6I','6J','6K','6L','6M',
             '6N','6O','6P','6Q','6R','6S','6T','6U','6V','6W','6X','6Y','6Z',
             '7A','7B','7C','7D','7E','7F','7G','7H','7I','7J','7K','7L','7M',
             '7N','7O','7P','7Q','7R','7S','7T','7U','7V','7W','7X','7Y','7Z']

    none_char = ' '
    num_sectors = 0
    sector_dict = {}

    print((2*shape[1]+2)*'-')
    for row in range(shape[0]):
        line = '|'
        for col in range(shape[1]):

            sector = matrix.tensor_coords_to_sector_coords([row,col])

            if sector is None:
                line += none_char.center(2,' ')
                continue

            if sector[0] not in sector_dict.keys():
                sector_dict[sector[0]] = symbols[num_sectors]
                num_sectors += 1

            line += sector_dict[sector[0]].center(2,' ')

        line += '|'
        print(line)
    print((2*shape[1]+2)*'-')

    print('\ncharacter | sector               | shape                | size')
    print(80*'-')
    for sector in sector_dict.keys():
        print('    ',sector_dict[sector],'   |', str(sector).ljust(20) ,'|',
              str(matrix.sectors[sector].shape).ljust(20),'|',
              matrix.sectors[sector].size)

def purify_tensor(tensor):
    '''
    Purifies the tensor by transforming all pipes into legs. The pipe structure
    will be lost after this.
    '''

    # The finding can be written shorter and prettier
    # but this does its work with low overhead.

    ndim = tensor.ndim

    # calculate the pure sectors
    pure_sectors = {}
    purified_sector_dict = get_sectors_to_be_joined_if_purified(tensor)
    for pure_sector_name in purified_sector_dict.keys():

        pure_sector_size = [0 for _ in range(ndim)]

        q_size = [{} for _ in range(ndim)]

        # find out sizes and names of subsectors
        for item in purified_sector_dict[pure_sector_name]:
            for n in range(ndim):
                if item[n] not in q_size[n]:
                    q_size[n][item[n]] = np.shape(tensor.sectors[item])[n]
                    pure_sector_size[n] += np.shape(tensor.sectors[item])[n]

        # find exact positions of subsectors in pure sector.
        q_pos = [{subsector:[] for subsector in q_size[n].keys()}
                 for n in range(ndim)]

        for n in range(ndim):
            counter = 0
            for entry in tensor.q_vectors[n]:
                if entry in q_size[n]:
                    q_pos[n][entry].append(counter)
                    counter += 1

        # construct pure sector
        pure_sectors[pure_sector_name] = np.zeros(pure_sector_size,
                                                  dtype=tensor.dtype)

        for subsector in itertools.product(*[n.keys() for n in q_pos]):

            grid = tuple(np.meshgrid(*[np.array(q_pos[n][entry])
                                       for n,entry in enumerate(subsector)],
                                     indexing='ij'))

            pure_sectors[pure_sector_name][grid] = tensor.sectors[subsector]

    #calculate pure charge vectors
    pure_charge_vectors = [purify_tuple(tensor.q_vectors[n])
                           for n in range(ndim)]
    
    return reducedTensor(tensor=pure_sectors,
                         list_of_q=copy.deepcopy(pure_charge_vectors),
                         list_of_xi=copy.deepcopy(tensor.q_signs),
                         Q=tensor.Q,sectors_given=True)

def _give_convenient_numpy_float_complex_type(type1,type2):
    '''
    Takes two data types which are either float{16,32,64,128} or
    complex{64,128,256} and gives back the best matching data type for both.
    The resulting type is small enough while preventing data loss.
    Example: type1 = np.float128, type2 = np.complex64, returns np.complex256.
    'complex' because one input was complex and '256' to fit float128 inside.
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

#######################
# reducedTensor class #
#######################

class reducedTensor:
    '''
    The reduced tensor class implements a class of sparse tensors where the
    potentially non-zero entries are organised into smaller tensors, called
    'charge sectors'. The position of an entry in a charge sector is mapped to
    a position in the larger tensor by vectors called 'charge vectors' and
    number called 'charge signs'. The charge signs may only be either +1 or -1.

    Each charge vector is associated with a dimension of the reduced tensor.
    Furthermore, the length of each charge vector equals the length of the axis
    it is associated with.

    An entry in the reducedTensor at position (a_{1},...,a_{n}) is allowed to
    be non-zero if the following equation holds

    \sum_{k=1}^{n} xi^{(k)} q^{(k)}_{a_{k}} = Q

    Here, n is the number of dimensions of tensor in question while xi^{(k)} is
    the charge signs for axis k. Furthermore, q^{(k)} is the charge vector
    associated with axis k and q^{(k)}_{a_{k}} is the a_{k}-th entry of exactly
    that vector. Q is the total charge of the vector.

    For every entry in the reduced tensor where the given equation does not
    hold, that entry must be zero. This means that the tensor in question is
    potentially very sparse resulting in the fact that it would be beneficial
    to store the tensor not in dense form but differently. Instead of using
    established one-size-fits-all methods, here we use a method custom-tailored
    to the requirements in question. Each entry (a_{1},...,a_{n}) satisfying
    the equation given above is written with all other entries also satisfying
    the equation given above as a tensor if the associated entries in the
    charge vectors q^{(1)}_{a_{1}},...,q^{(1)}_{a_{1}} for all those entries
    match. In this sense, all potentially non-zero entries of the tensor get
    sorted into smaller tensors which eventually are stored in the
    reducedTensor object and on which all calculations are performed.

    More information about reduced tensors may be found in chapter 5 of
    doi: 10.21468/SciPostPhysLectNotes.5 (arXiv:1805.00055)
    '''

    def __init__(self,tensor,list_of_q,list_of_xi,Q=0,dtype=None,*,
                 sectors_given=False,
                 pipeID=None,pipehighestID=None,pipe_dict=None):
        '''
        Initialises the reducedTensor object. There are several ways to do
        this. The argument tensor may either be given as a np.ndarray which is
        then decayed into its charge sectors or as a dictionary where the keys
        are charge sector addresses and the values the corresponding charge
        sectors. In the latter case sectors_given must be set to False, while
        in the former case it must be set to True.

        In both cases, charge vectors (list_of_q) charge signs (list_of_xi) and
        a total charge (Q) must be supplied.

        In addition to this, it is also possible to set the dtype of the charge
        sectors, which are numpy arrays. Default is None, causing the
        initialisation function to determine the most appropriate dtype
        automatically. Another possible value is 'ref' causing the given charge
        sectors to remain unmodified.

        It is also possible to provide a pipe infrastructure managing the pipes
        and subsectors of the reduced tensor if required. This may be done by
        setting pipeID, pipehighestID, pipe_dict accordingly.
        '''

        # set charge attributes
        self.Q         = Q
        self.q_vectors = [list(entry) for entry in list_of_q]
        self.q_signs   = list_of_xi

        if sectors_given:

            # charge sectors are given

            self.sectors = {}
            self.ndim    = len(self.q_vectors)
            self.shape   = tuple([len(q) for q in self.q_vectors])
            self.size    = np.prod(self.shape)

            if dtype == 'ref':
                # The sectors are passed by reference.
                # No modification will occur.
                for key,value in tensor.items():
                    self.sectors = tensor

                # Take the dtype from some sector.
                # Note: sectors should not have mixed dtype for this.
                if len(tensor.keys()) == 0:
                    self.dtype = np.float64
                else:
                    self.dtype = tensor[list(tensor.keys())[0]].dtype
            else:
                # The sectors are modified to have the correct dtype.
                if dtype is None:
                    dtype = np.float16
                    for sector in tensor.values():
                        dtype = _give_convenient_numpy_float_complex_type(
                            dtype,sector.dtype.type)

                self.dtype = np.dtype(dtype)
                for key,value in tensor.items():
                    self.sectors[key] = value.astype(self.dtype)

            self.filling = (np.sum([np.size(self.sectors[i])
                                    for i in self.sectors.keys()]))

            self.density = self.filling / self.size

        else:

            # tensor is given
            
            self.sectors = {}
            self.ndim    = np.ndim(tensor)
            self.shape   = np.shape(tensor)
            self.size    = np.size(tensor)
            self.filling = 0  # \__ will be set later
            self.density = 0. # /

            if dtype is None:
                self.dtype = tensor.dtype
                self._split_tensor_in_sectors(tensor)
            else:
                self.dtype = (np.dtype(dtype) if not np.issubdtype(
                    dtype, np.integer) else np.dtype(np.float64))

                self._split_tensor_in_sectors(tensor.astype(self.dtype))

        # infrastructure for leg binding
        if pipeID is None:
            self.pipeID = self.ndim*[None] # ID of pipes. Legs get None.
        else:
            self.pipeID = pipeID

        if pipe_dict is None:
            self.pipe_dict = {} # In the beginning, there were no pipes.
        else:
            self.pipe_dict = pipe_dict

        if pipehighestID is None:
            self.pipehighestID = 0 # Used to supply a new ID collision free.
        else:
            self.pipehighestID = pipehighestID

    def _split_tensor_in_sectors(self,tensor):
        '''
        This function takes the given tensor which is of type np.ndarray and
        slices it into charge sectors according to the information acquired
        during the initialisation. If entries of 'tensor' are non-zero where
        such non-zero entries are not allowed, they will not be stored and will
        therefore be lost.
        '''

        dimension = tensor.ndim

        list_of_s = []
        num_sectors = 1

        # create the unique q_{i}
        for q in self.q_vectors:
            list_of_s.append(list(set(q)))
            num_sectors *= len(list_of_s[-1])

        # iterate through all sectors
        for sector_name in itertools.product(*list_of_s):

            # find out if sector is valid
            if np.dot(purify_tuple(sector_name),self.q_signs) != self.Q:
                continue

            # evaluate positions of the sector
            f = []
            for i in range(dimension):
                f.append([])
                for j in range(len(self.q_vectors[i])):
                    if sector_name[i] == self.q_vectors[i][j]:
                        f[-1].append(j)

            # carve sector out of tensor and store it
            self.sectors[sector_name]=tensor[tuple(np.meshgrid(*f,
                                                               indexing='ij'))]
            self.filling += np.size(self.sectors[sector_name])

        self.density = self.filling / self.size

    def tensor_coords_to_sector_coords(self,coords):
        '''
        Takes a set of coordinates associated with the entire tensor and
        translates them to the coordinates of a specific tensor. Returns the
        charge sector address along with the coordinates within that sector the
        global coordinates refer to.
        '''

        sector = tuple([self.q_vectors[i][coords[i]]
                        for i in range(len(coords))])

        if sector not in self.sectors.keys():
            return None

        q_vector_reduced = [self.q_vectors[i][:coords[i]+1]
                            for i in range(len(coords))]

        sector_coords=tuple([q_vector_reduced[i].count(q_vector_reduced[i][-1])
                             - 1 for i in range(len(coords))])

        return sector, sector_coords

    def get_element_in_tensor_coords(self,coords):
        '''
        Returns an element of the reduced tensor whose position is given in
        global coordinates. If the coordinates refer to a position outside any
        charge sector, 0.0 is returned.
        '''

        sector = self.tensor_coords_to_sector_coords(coords)
        if sector is None:
            return 0.0

        return self.sectors[sector[0]][sector[1]]

    def set_element_in_tensor_coords(self,coords,value,dontRaiseError = False):
        '''
        Attempts to set an entry of the reduced tensor whose position is given
        by 'coords' in global coordinates to the value provided in 'value'. If
        this is not possible because the given position lies outside any
        present charge sector, an error is raised if dontRaiseError=False and
        the value 1 is returned if dontRaiseError=True. For the case that the
        value can be set without incident, the value 0 is returned.

        TODO: Also raises error if some sectors simply do not exist but would
        be valid. In this case, the missing valid sector has to be created and
        the respective value being set while all other values must be zero.
        '''

        sector = self.tensor_coords_to_sector_coords(coords)

        if sector is None:
            if not dontRaiseError:
                raise IndexError('Given coordinates lie '
                                 'outside of valid sectors.')
            else:
                return 1

        self.sectors[sector[0]][sector[1]] = value
        return 0

    def list_sectors(self):
        '''
        Returns a list of all sectors currently present in the reduced tensor.
        '''

        return list(self.sectors.keys())

    def reconstruct_full_tensor(self):
        '''
        Reconstructs the full tensor in dense form. Depending on the size of
        the reduced tensor, a significant amount of storage may be required.
        '''

        tensor = np.empty(self.shape,dtype=self.dtype)
        loop_set = [np.arange(i) for i in self.shape]

        for coords in itertools.product(*loop_set):
            tensor[coords] = self.get_element_in_tensor_coords(coords)

        return tensor

    def swapaxes(self,axis1,axis2):
        '''
        Swaps the supplied axes of this tensor inplace.             
        '''
        
        return swapaxes_for_reduced_tensor(self,axis1,axis2)

    def conjugate(self):
        '''
        Conjugates the reduced tensor by conjugating all numbers stored in the
        charge sectors. In addition, the charge signs and the total charge is
        flipped.
        '''

        # conjugate all sectors
        for sector in self.sectors.keys():
            self.sectors[sector] = self.sectors[sector].conjugate()

        # flip charge signs
        self.q_signs = list(-np.array(self.q_signs))

        # flip tensor charge
        self.Q = -self.Q

        return self

    def abs_min(self):
        '''
        Gives the value of the lowest magnitude stored in the sectors.
        Zeros which are not saved are not considered.
        '''

        am = np.inf

        for sector in self.sectors.keys():
            am = np.min([am,np.min(np.abs(self.sectors[sector]))])
            print(am)

        return am

    def lowest_magnitude(self):
        '''
        Gives the value of the lowest magnitude stored in the sectors.
        Zeros which are not saved are not considered. It also ignores zeros
        which are saved. This is the only difference to 'abs_min'.
        '''

        am = np.inf

        for sector in self.sectors.keys():
            a = self.sectors[sector]
            nonzero = a[np.nonzero(a)]
            if nonzero.size == 0:
                return -np.inf

            am = np.min([am,np.min(np.abs(nonzero))])

        return am if not np.isinf(am) else -np.inf

    def astype(self,dtype):
        '''
        Returns a copy of the reduced tensor where each sector has been recast
        into the given dtype.
        '''

        tensor2 = copy.deepcopy(self)

        for key in tensor2.sectors.keys():
            tensor2.sectors[key] = tensor2.sectors[key].astype(dtype)

        return tensor2

    def print_tensor_info(self, print_vector_info = False,
                          print_sector_info = False, print_basis_info = True):
        '''
        Prints information about the reduced tensor to the standard output with
        varying degrees of verbosity.

        The following arguments are available:

        print_vector_info : Prints information regarding the charge vectors and
                            charge signs. Default is False.
        print_sector_info : Prints information about all present charge
                            sectors. Default is False.
        print_basis_info  : Prints the most essential information.
                            Default is True.
        '''

        if print_basis_info:
            filling = 10
            print(str('ndim').ljust(filling),    self.ndim)
            print(str('shape').ljust(filling),   self.shape)
            print(str('size').ljust(filling),    self.size)
            print(str('sectors').ljust(filling), len(self.sectors.keys()))
            print(str('filling').ljust(filling), self.filling)
            print(str('density').ljust(filling), self.density)
            print(str('Q').ljust(filling),       self.Q)

        if print_vector_info:
            if print_basis_info:
                print()
            print('dim | sign | vector')
            print(80*'-')
            for i in range(len(self.q_vectors)):
                sign = '  + ' if self.q_signs[i] == 1 else '  - '

                print(str(i).ljust(3),'|',sign,'|',self.q_vectors[i])

        if print_sector_info:
            if print_vector_info or print_basis_info:
                print()
            print('sector               | shape                | size')
            print(80*'-')
            for key in self.sectors.keys():
                print(str(key).ljust(20),'|',
                      str(np.shape(self.sectors[key])).ljust(20),'|',
                      np.size(self.sectors[key]))


    def change_dtype(self,dtype):
        '''
        Change the dtype of the reduced tensor by recasting all charge sectors
        to the new dtype inplace.
        '''

        if np.issubdtype(dtype, np.integer):
            raise TypeError('An integer type may not be used as datatype for '
                            'a reducedTensor. Only float or complex are '
                            'allowed.')

        self.dtype = np.dtype(dtype)

        for key,value in self.sectors.items():
            self.sectors[key] = value.astype(self.dtype)
