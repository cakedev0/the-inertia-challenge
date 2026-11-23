# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False
# The functions share no state, and their loops already run in parallel with
# the GIL released: they need no GIL on a free-threaded Python.
# cython: freethreading_compatible=True
"""Loops of the scorer on a chunk of rows.

The sums of the clusters of all the arrays of labels are the rows of one array,
`sums`: the ones of array `a` start at row `offsets[a]`.

A call only adds the rows of the clusters of its part, the ones with
`label % n_parts == part`. So the calls of the parts of a chunk can run at the
same time, they write to different rows, and each cluster gets its rows in
order whatever the number of parts.
"""

cdef extern from *:
    """
    #if defined(__GNUC__) || defined(__clang__)
    #define prefetch_for_write(address) __builtin_prefetch((address), 1, 3)
    #else
    #define prefetch_for_write(address) ((void)0)
    #endif
    """
    void prefetch_for_write(const void* address) nogil



def add_dense(
    const float[:, ::1] X,
    const unsigned int[:, ::1] labels,
    const long long[::1] offsets,
    double[:, ::1] sums,
    long long[::1] counts,
    unsigned int part,
    unsigned int n_parts,
):
    """Add each row of `X` to the sum and the count of its cluster, for each
    array (row) of `labels`. Return False if a label is not one of a cluster."""
    cdef Py_ssize_t n_rows = X.shape[0], n_features = X.shape[1]
    cdef Py_ssize_t n_arrays = labels.shape[0]
    cdef Py_ssize_t i, j, a
    cdef long long cluster
    cdef unsigned int label
    cdef const float* row
    cdef double* total
    cdef bint valid = True
    # With many clusters, most of the time goes to cache misses on `sums`:
    # asking for the sums of the row after the next one makes the loop ~15%
    # faster on deep_100M.
    cdef Py_ssize_t ahead = 2

    if n_rows == 0 or n_features == 0:
        return True
    with nogil:
        for i in range(n_rows):
            if i + ahead < n_rows:
                for a in range(n_arrays):
                    label = labels[a, i + ahead]
                    cluster = offsets[a] + label
                    if label % n_parts == part and cluster < offsets[a + 1]:
                        total = &sums[cluster, 0]
                        for j in range(0, n_features, 8):
                            prefetch_for_write(total + j)
                        prefetch_for_write(total + n_features - 1)
            row = &X[i, 0]
            for a in range(n_arrays):
                label = labels[a, i]
                if label % n_parts != part:
                    continue
                cluster = offsets[a] + label
                if cluster >= offsets[a + 1]:
                    valid = False
                    continue
                counts[cluster] += 1
                total = &sums[cluster, 0]
                for j in range(n_features):
                    total[j] += row[j]
    return valid


def add_sparse(
    const float[::1] data,
    const long long[::1] indices,
    const long long[::1] indptr,
    const unsigned int[:, ::1] labels,
    const long long[::1] offsets,
    double[:, ::1] sums,
    long long[::1] counts,
    unsigned int part,
    unsigned int n_parts,
):
    """`add_dense` for the rows of a CSR matrix."""
    cdef Py_ssize_t n_rows = indptr.shape[0] - 1
    cdef Py_ssize_t n_arrays = labels.shape[0]
    cdef Py_ssize_t i, a
    cdef long long cluster, p
    cdef unsigned int label
    cdef bint valid = True

    with nogil:
        for i in range(n_rows):
            for a in range(n_arrays):
                label = labels[a, i]
                if label % n_parts != part:
                    continue
                cluster = offsets[a] + label
                if cluster >= offsets[a + 1]:
                    valid = False
                    continue
                counts[cluster] += 1
                for p in range(indptr[i], indptr[i + 1]):
                    sums[cluster, indices[p]] += data[p]
    return valid


def squared_norms_dense(const float[:, ::1] X, double[::1] out):
    """Write the squared norm of each row of `X` to `out`."""
    cdef Py_ssize_t n_rows = X.shape[0], n_features = X.shape[1]
    cdef Py_ssize_t i, j
    cdef double value
    cdef double[4] norms

    with nogil:
        for i in range(n_rows):
            # Four sums, for the processor to run them at the same time.
            norms[0] = norms[1] = norms[2] = norms[3] = 0.0
            for j in range(n_features):
                value = X[i, j]
                norms[j & 3] += value * value
            out[i] = (norms[0] + norms[1]) + (norms[2] + norms[3])


def squared_norms_sparse(
    const float[::1] data, const long long[::1] indptr, double[::1] out
):
    """`squared_norms_dense` for the rows of a CSR matrix."""
    cdef Py_ssize_t n_rows = indptr.shape[0] - 1
    cdef Py_ssize_t i
    cdef long long p
    cdef double value, norm

    with nogil:
        for i in range(n_rows):
            norm = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                value = data[p]
                norm += value * value
            out[i] = norm
