import numpy as np

def optimize_vector_sum(vector_library, target, error_bound = 1e-5):
    error = target
    vector_sum = []

    vector_library_copy = vector_library.copy()

    while vector_library_copy.any() and np.linalg.norm(error) > error_bound:
        alignments = vector_library_copy @ error
        most_aligned_idx = np.argmax(abs(alignments))
        most_aligned_vector = vector_library[most_aligned_idx]

        scaling_coeff = np.dot(most_aligned_vector, error) / np.dot(most_aligned_vector, most_aligned_vector)
        vector_sum.append((most_aligned_idx, scaling_coeff))

        error = error - scaling_coeff * most_aligned_vector
        vector_library_copy[most_aligned_idx] = np.zeros_like(most_aligned_vector)

    final_vector = np.zeros_like(error)
    for vector_idx, coeff in vector_sum:
        final_vector += coeff * vector_library[vector_idx]

    return vector_sum, final_vector, error

def optimize_constrainted_vector_sum(vector_library, target, covariance_matrix, idio_matrix, error_bound = 1e-5, variance_bias = .75):
    error = np.array(target)
    vector_sum = np.zeros(len(vector_library))

    vector_library_copy = vector_library.copy()
    c = 0

    while vector_library_copy.any() and np.linalg.norm(error) > (error_bound * np.exp(c * error_bound ** (1/2))):
        all_coeffs = vector_library_copy @ error / (np.einsum("ij,ij->i", vector_library_copy, vector_library_copy) + 1e-8)
        possible_weights = np.array([vector_sum for i in all_coeffs]) + np.diag(all_coeffs)

        alignments = -abs(all_coeffs) / (error.T @ error) ** (1/2) * (np.einsum("ij,ij->i", vector_library_copy, vector_library_copy) + 1e-8) ** (1/2)
        variance = np.diag(possible_weights.T @ vector_library @ covariance_matrix @ (possible_weights.T @ vector_library).T + possible_weights.T @ idio_matrix @ possible_weights) / np.sum(abs(possible_weights), axis = 1)**2
        
        bias = variance_bias * np.exp(-c / len(vector_sum))
        loss = alignments * (1 - bias) + variance * bias

        best_idx = np.argmin(loss)

        best_vector = vector_library[best_idx]
        scaling_coeff = all_coeffs[best_idx]

        vector_sum[best_idx] += scaling_coeff
        error = error - scaling_coeff * best_vector

        c += 1

    final_vector = vector_sum.T @ vector_library
    final_variance = vector_sum.T @ vector_library @ covariance_matrix @ (vector_sum.T @ vector_library).T + vector_sum.T @ idio_matrix @ vector_sum / np.sum(abs(possible_weights))**2

    return vector_sum / np.sum(abs(vector_sum)), final_vector, error, final_variance / np.sum(abs(vector_sum)) ** 2 * 252, c