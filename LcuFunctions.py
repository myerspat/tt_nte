import numpy as np
import math

from scipy.linalg import expm

from qiskit.circuit import QuantumCircuit


# Perform Gram-Schmidt orthogonalization to return V vector from LCU paper.
# The first column is the normalized set of sqrt(alpha) values,
# all other columns' actual values are irrelevant to the algorithm but are set to be
# orthogonal to the alpha vector to ensure V is unitary.
# Function modified from ChatGPT suggestion, O(2^(2L))
def gram_schmidt_ortho(vector):
    num_dimensions = len(vector)
    ortho_basis = np.zeros((num_dimensions, num_dimensions), dtype=float)
    
    # Normalize the input vector and add it to the orthogonal basis
    ortho_basis[0] = vector / np.linalg.norm(vector)
    #print(ortho_basis)

    # Gram-Schmidt orthogonalization
    for i in range(1, num_dimensions):
        ortho_vector = np.random.rand(num_dimensions)  # random initialization
        #if(abs(np.log2(i) - np.ceil(np.log2(i))) < 0.00001):
        #    print("dimension: ", i)
        for j in range(i):
            ortho_vector -= np.dot(ortho_basis[j], ortho_vector) / np.dot(ortho_basis[j], ortho_basis[j]) * ortho_basis[j]
        ortho_basis[i] = ortho_vector / np.linalg.norm(ortho_vector)
    
    return ortho_basis.T


# return True if matrix is unitary, False otherwise, O(len(matrix)^2)
def is_unitary(matrix):
    I = matrix.dot(np.conj(matrix).T)
    return I.shape[0] == I.shape[1] and np.allclose(I, np.eye(I.shape[0]))


def get_b_setup_gate(vector, nb):
    if isinstance(vector, list):
        vector = np.array(vector)
    vector_circuit = QuantumCircuit(nb)
    vector_circuit.isometry(
        vector / np.linalg.norm(vector), list(range(nb)), None
    )
    return vector_circuit


# return the U gate in the LCU process as well as a vector of the alpha values.
# Use the Fourier process from the LCU paper to approximate the inverse of A, O(2^L * N^2) ??, need to verify this
def get_fourier_unitaries(J, K, y_max, z_max, matrix, doFullSolution, A_mat_size):
    A_mat_size = len(matrix)
    delta_y = y_max / J
    delta_z = z_max / K
    if doFullSolution:
        U = np.zeros((A_mat_size * 2 * J * K, A_mat_size * 2 * J * K), dtype=complex) # matrix from 
        alphas = np.zeros(2 * J * K)
    M = np.zeros((A_mat_size, A_mat_size)) # approximation of inverse of A matrix

    for j in range(J):
        y = j * delta_y
        for k in range(-K,K):
            z = k * delta_z
            alpha_temp = (1) / math.sqrt(2 * math.pi) * delta_y * delta_z * z * math.exp(-z*z/2)
            uni_mat = (1j) * expm(-(1j) * matrix * y * z)
            assert(is_unitary(uni_mat))
            M_temp = alpha_temp * uni_mat
            if doFullSolution:
                if(alpha_temp < 0): # if alpha is negative, incorporate negative phase into U unitary
                    alpha_temp *= -1
                    U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = -1 * uni_mat
                else:
                    alpha_temp *= 1
                    U[A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1),A_mat_size*(2 * j * K + (k + K)):A_mat_size*(2 * j * K + (k + K) + 1)] = uni_mat
                alphas[2 * j * K + (k + K)] = alpha_temp
            M = M + M_temp

    matrix_invert = np.linalg.inv(matrix)
    error_norm = np.linalg.norm(M - matrix_invert)
    if doFullSolution:
        #print("real matrix inverse: ", matrix_invert)
        #print("probability of algorithm success: ", math.pow(np.linalg.norm(np.matmul(matrix, vector/ np.linalg.norm(vector))),2))
        #print("estimated matrix inverse: ", M)
        #print("Matrix inverse error: ", (M - matrix_invert) / matrix_invert)
        
        #print("norm of inverse error: ", error_norm)

        return U, alphas, error_norm
    return 0, 0, error_norm
