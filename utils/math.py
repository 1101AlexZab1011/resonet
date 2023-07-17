import torch

def mode_n_flattening(arr, mode):
    """
    Perform mode-n flattening on a PyTorch tensor.

    Parameters:
    - arr (Tensor): Input PyTorch tensor.
    - mode (int): The mode to flatten along (0-based index).

    Returns:
    - flattened_arr (Tensor): Flattened tensor.
    """
    if mode < 0 or mode >= arr.ndim:
        raise ValueError('Invalid mode value.')

    # Permute the mode axis to the front
    permuted_arr = arr.permute(mode, *range(mode), *range(mode+1, arr.ndim))

    # Reshape the tensor
    flattened_arr = permuted_arr.reshape(permuted_arr.size(0), -1)

    return flattened_arr


import torch

def mode_n_product(x, m, mode):
    """
    Perform mode-n product (matrix multiplication) on a PyTorch tensor.

    Parameters:
    - x (Tensor): Input PyTorch tensor.
    - m (Tensor): Matrix to multiply with.
    - mode (int): The mode to multiply along (0-based index).

    Returns:
    - result (Tensor): Result of the mode-n product.
    """
    if mode < 0:
        raise ValueError('`mode` must be a positive integer')
    if x.ndim < mode:
        raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
    if m.ndim != 2:
        raise ValueError('Invalid shape of M: {}'.format(m.shape))

    # Permute the mode axis to the front
    x = x.permute(*list(range(mode)) + list(range(mode + 1, x.ndim)) + [mode])

    # Perform matrix multiplication
    result = torch.matmul(x, m.T)

    # Permute the mode axis back to its original position
    result = result.permute(*list(range(mode)) + [result.ndim - 1] + list(range(mode, result.ndim - 1)))

    return result

def hosvd(A):
    """
    Perform Higher Order Singular Value Decomposition (HOSVD) on a NumPy array.

    Parameters:
    - A (ndarray): Input NumPy array.

    Returns:
    - S (ndarray): Core tensor after HOSVD.
    - U (list): List of orthogonal matrices U (factors) after HOSVD.
    """
    U = []

    for i in range(A.ndim):
        Am = mode_n_flattening(A, i)
        Um, _, _ = torch.linalg.svd(Am, full_matrices=True)
        U.append(Um)

    S = A.clone()

    for i in range(A.ndim):
        S = mode_n_product(S, U[i].T, i)

    return S, U
