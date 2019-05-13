import numpy as np
from itertools import permutations, combinations
from collections import deque
from functools import reduce

#################################################################
# Group Definitions
#################################################################


def orth_basis(rank):
    """Generates an orthonormal basis of zeros and ones of the same dimension
    as the rank.

    :param rank: the dimension of the vectors we will use
    :return: a rank * rank matrix with the ith row and column being one

    >>> orth_basis(3)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])
    """
    # this will be the building block of all subsequent computations
    return np.identity(rank, dtype=int)

# Main Group Functions


class Group:
    """The instance functions for the group that are universal.
    These are mostly to do with manipulating the simple roots
    and positive roots. All other things are generated from this.

    Do not use this class for calculations. It is simply a store for
    all common functions.
    """
    def __init__(self, rank):
        self.rank = rank

    def get_rank(self):
        return self.rank

    def simple_roots(self, basis='dynkin'):
        """This will return the simple roots in a nice way for
        comprehension.
        """
        if basis.lower() == 'dynkin':
            mat = self._simple_roots()
            return [list(mat[i, :]) for i in range(self.rank)]

        elif basis.lower() == 'orthogonal':
            mat = self._simple_roots(basis='orthogonal')
            return [list(mat[i, :]) for i in range(self.rank)]

        elif basis.lower() == 'alpha':
            mat = self._simple_roots(basis='alpha')
            return [list(mat[i, :]) for i in range(self.rank)]

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def positive_roots(self, basis='dynkin'):
        """This will return the simple roots in a nice way for
        comprehension.

        Note that this is not necessarily ordered, i.e. the simple
        roots do not appear first.
        """
        if basis.lower() == 'dynkin':
            mat = self._positive_roots()
            return [list(mat[i, :]) for i in range(len(mat))]

        elif basis.lower() == 'orthogonal':
            mat = self._positive_roots(basis='orthogonal')
            return [list(mat[i, :]) for i in range(len(mat))]

        # note that this doesn't work for B, C, D yet.
        elif basis.lower() == 'alpha':
            mat = self._positive_roots(basis='alpha')
            return [list(mat[i, :]) for i in range(len(mat))]

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def cartan_matrix(self):
        """Returns the Cartan matrix."""
        return self._simple_roots()

    def quadratic_form(self):
        """This creates the quadtratic form, which is the metric in
        the Dynkin basis.
        """
        # The so called D matrix:
        # D = Diag[1/2 * (<alpha_1,alpha_1> + ...  + <alpha_r, alpha_r>)]
        D = np.diag([1/2 * np.dot(
            self._simple_roots(basis='orthogonal')[i, :],
            self._simple_roots(basis='orthogonal')[i, :])
            for i in range(self.rank)])

        cartan_inv = np.linalg.inv(self._simple_roots())
        return np.dot(cartan_inv, D)

    def inner_product(self, vec1, vec2, basis='dynkin'):
        """Returns the inner products in the relevant bases."""
        if basis == 'dynkin':
            quadratic_form = self.quadratic_form()
            return np.dot(np.dot(vec1, quadratic_form), vec2)

        elif basis == 'orthogonal':
            return np.dot(vec1, vec2)

        else:
            print('Please use the orthogonal basis or the dynkin basis.')

    def weyl_vector(self):
        """Returns the half sum of the positive roots."""

        rho = np.sum(self._positive_roots(), axis=0)/2
        return rho

    def basis_changer(self, vector, basis1, basis2):
        """Takes a vector in one of the three bases, and changes it into one
        the other three bases.

        :param vector: the vector itself
        :param basis1: the basis the vector starts in. Options are "alpha",
        "orthogonal" or "dynkin".
        :param basis2: the basis the vector ends in. Options are "alpha",
        "orthogonal" or "dynkin".

        >>> orth_basis(3)
        array([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
        """
        if basis1 == 'alpha':
            if basis2 == 'alpha':
                return vector
            elif basis2 == 'orthogonal':
                return self._alpha_to_orthogonal(vector)
            elif basis2 == 'dynkin':
                return self._alpha_to_dynkin(vector)
            else:
                print("""The second basis needs to be one of \"alpha\",
                      \"orthogonal\" or \"dynkin\".""")
                return

        elif basis1 == 'orthogonal':
            if basis2 == 'alpha':
                return self._orthogonal_to_alpha(vector)
            elif basis2 == 'orthogonal':
                return vector
            elif basis2 == 'dynkin':
                return self._orthogonal_to_dynkin(vector)
            else:
                print("""The second basis needs to be one of \"alpha\",
                      \"orthogonal\" or \"dynkin\".""")
                return

        elif basis1 == 'dynkin':
            if basis2 == 'alpha':
                return self._dynkin_to_alpha(vector)
            elif basis2 == 'orthogonal':
                return self._dynkin_to_orthogonal(vector)
            elif basis2 == 'dynkin':
                return vector
            else:
                print("""The second basis needs to be one of \"alpha\",
                      \"orthogonal\" or \"dynkin\".""")
                return

        else:
            print("""The first basis needs to be one of \"alpha\",
                  \"orthogonal\" or \"dynkin\".""")
            return

    def reflection(self, i):
        """Return a Weyl reflection function zero-indexed by i."""
        roots = self._simple_roots(basis='orthogonal')
        return (lambda vec: np.array(vec)
                - 2 * np.dot(np.array(vec), roots[i])
                / np.dot(roots[i], roots[i]) * roots[i]
                if len(vec) == self.rank else 'Ranks do not match.')

    def weyl_reflections(self):
        """Here we generate a list of functions that act on vectors.
        These functions are the shifted Weyl reflections associated
        with the positive roots.
        """
        return [self.reflection(i) for i in range(self.rank)]

    def shifted_reflection(self, i):
        """Return a Weyl reflection of a weight shifted by rho function
        zero-indexed by i.
        """
        rho = self.weyl_vector()
        roots = self._positive_roots()
        return (lambda vec: np.array(vec)
                - 2 * self.inner_product(np.array(vec) + rho, roots[i])
                / self.inner_product(roots[i], roots[i]) * roots[i]
                if len(vec) == self.rank else 'Ranks do not match.')

    def racah_reflections(self):
        """Here we generate a list of functions that act on vectors.
        These functions are the shifted Weyl reflections associated
        with the positive roots.
        """
        num = len(self._positive_roots())
        return [self.shifted_reflection(i) for i in range(num)]

    def _simple_roots(self, basis='dynkin'):
        """Placeholder function. The real functions are in the subclasses
        for ABCD groups.
        """
        print('Specify an ABCD type group first.')
        return np.array([[]])

    def _positive_roots(self, basis='dynkin'):
        """Placeholder function. The real functions are in the subclasses
        for ABCD groups.
        """
        print('Specify an ABCD type group first.')
        return np.array([[]])

    def _alpha_to_dynkin(self, basis='dynkin'):
        """Placeholder function. The real functions are in the subclasses
        for ABCD groups.
        """
        print('Specify an ABCD type group first.')
        return np.array([[]])

    def _alpha_to_orthogonal(self, basis='dynkin'):
        """Placeholder function. The real functions are in the subclasses
        for ABCD groups.
        """
        print('Specify an ABCD type group first.')
        return np.array([[]])

    def _dynkin_to_alpha(self, basis='dynkin'):
        """Placeholder function. The real functions are in the subclasses
        for ABCD groups.
        """
        print('Specify an ABCD type group first.')
        return np.array([[]])

    def _dynkin_to_orthogonal(self, basis='dynkin'):
        """Placeholder function. The real functions are in the subclasses
        for ABCD groups.
        """
        print('Specify an ABCD type group first.')
        return np.array([[]])

    def _orthogonal_to_alpha(self, basis='dynkin'):
        """Placeholder function. The real functions are in the subclasses
        for ABCD groups.
        """
        print('Specify an ABCD type group first.')
        return np.array([[]])

    def _orthogonal_to_dynkin(self, basis='dynkin'):
        """Placeholder function. The real functions are in the subclasses
        for ABCD groups.
        """
        print('Specify an ABCD type group first.')
        return np.array([[]])

# A_N / SU(N+1) Groups


class AGroup(Group):
    """Data for the A_N or SU(N+1) representations. The rank needs to be a
    positive integer.
    """

    def __new__(cls, rank):
        """The group is only initiated with a valid rank."""
        if rank > 0 and type(rank) == int:
            return super(AGroup, cls).__new__(cls)

        else:
            raise Exception('The rank needs to be a positive integer')

    def __init__(self, rank):
        super().__init__(rank)

    def __repr__(self):
        return 'A' + str(self.rank)

    def _simple_roots(self, basis='dynkin'):
        """The matrix form of the simple roots for calculations."""
        simple_roots = np.zeros((self.rank, self.rank + 1), dtype=int)
        for i in range(self.rank):
            simple_roots[i, :] = (orth_basis(self.rank + 1)[i, :]
                                  - orth_basis(self.rank + 1)[i + 1, :])

        if basis.lower() == 'dynkin':
            return np.array([self._orthogonal_to_dynkin(root) for root in
                             simple_roots])

        elif basis.lower() == 'orthogonal':
            return simple_roots

        elif basis.lower() == 'alpha':
            return np.array([self._orthogonal_to_alpha(root) for root in
                             simple_roots])

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def _positive_roots(self, basis='dynkin'):
        """The matrix form of the positive roots for calculations."""
        positive_roots = np.zeros((self.rank*((self.rank) + 1)//2,
                                  self.rank + 1), dtype=int)

        k = 0
        for i in range(0, self.rank):
            for j in range(i+1, self.rank + 1):
                positive_roots[k, :] = (orth_basis(self.rank + 1)[i, :]
                                        - orth_basis(self.rank + 1)[j, :])
                k += 1

        if basis.lower() == 'dynkin':
            return np.array([self._orthogonal_to_dynkin(root) for root in
                             positive_roots])

        elif basis.lower() == 'orthogonal':
            return positive_roots

        elif basis.lower() == 'alpha':
            return np.array([self._orthogonal_to_alpha(root) for root in
                             positive_roots])

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def _orthogonal_to_dynkin(self, vector):
        """Is a function that acts on a vector in the orthogonal basis.
        This function converts from the orthogonal to Dynkin basis.
        """
        return np.array([int(round(vector[i] - vector[i + 1]))
                        for i in range(self.rank)])

    def _orthogonal_to_alpha(self, vector):
        """Acts on a vector in the orthogonal basis. It converts it into
        a vector in the alpha basis.
        """
        new_vec = []
        new_val = 0
        for i in range(len(vector) - 1):
            new_val += vector[i]
            new_vec.append(new_val)
        return new_vec

    def _dynkin_to_orthogonal(self, vector):
        """Acts on a vector in the Dynkin basis.
        This converts from the Dynkin basis to the orthogonal basis.
        """
        new_vec = []
        new_val = 0
        for i in range(self.rank):
            new_val += (self.rank - i) * vector[i] / (self.rank + 1)

        new_vec.append(new_val)
        for i in range(self.rank):
            new_val -= vector[i]
            new_vec.append(new_val)
        return new_vec

    def _dynkin_to_alpha(self, vector):
        """We simply convert the Dynkin into orthogonal, and then orthogonal
        into alpha basis.
        """
        return self._orthogonal_to_alpha(self._dynkin_to_orthogonal(vector))

    def _alpha_to_dynkin(self, vector):
        """Converts the alpha basis into the Dynkin basis. This is much easier
        with the transpose of the Cartan matrix.
        """
        return list(np.dot(self.cartan_matrix().T, vector))

    def _alpha_to_orthogonal(self, vector):
        """Alpha to Dynkin, then Dynkin to orthogonal.
        """
        return self._dynkin_to_orthogonal(self._alpha_to_dynkin(vector))


# B_N / SO(2N+1) Groups


class BGroup(Group):
    """Data for the B_N or SO(2N+1) representations. The rank needs to be a positive integer.
    """

    def __new__(cls, rank):
        """The group is only initiated with a valid rank."""
        if rank > 0 and type(rank) == int:
            return super(BGroup, cls).__new__(cls)

        else:
            raise Exception('The rank needs to be a positive integer')

    def __init__(self, rank):
        super().__init__(rank)

    def __repr__(self):
        return 'B' + str(self.rank)

    def _simple_roots(self, basis='dynkin'):
        """The matrix form of the simple roots for calculations."""
        simple_roots = np.zeros((self.rank, self.rank), dtype=int)
        for i in range(self.rank):
            if i == self.rank - 1:
                simple_roots[i, :] = orth_basis(self.rank)[i, :]
            else:
                simple_roots[i, :] = (orth_basis(self.rank)[i, :]
                                      - orth_basis(self.rank)[i + 1, :])

        if basis.lower() == 'dynkin':
            return np.array([self._orthogonal_to_dynkin(row) for row in
                             simple_roots])

        elif basis.lower() == 'orthogonal':
            return simple_roots

        elif basis.lower() == 'alpha':
            return np.array([self._orthogonal_to_alpha(row) for row in
                             simple_roots])

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def _positive_roots(self, basis='dynkin'):
        """The matrix form of the positive roots for calculations."""
        positive_roots = np.zeros(((self.rank)*(self.rank), self.rank),
                                  dtype=int)

        k = 0
        for i in range(0, self.rank):
            for j in range(i, self.rank):
                if j == i:
                    positive_roots[k, :] = orth_basis(self.rank)[i, :]
                    k += 1
                else:
                    positive_roots[k, :] = (orth_basis(self.rank)[i, :]
                                            + orth_basis(self.rank)[j, :])
                    k += 1
                    positive_roots[k, :] = (orth_basis(self.rank)[i, :]
                                            - orth_basis(self.rank)[j, :])
                    k += 1

        if basis.lower() == 'dynkin':
            return np.array([self._orthogonal_to_dynkin(root) for root in
                             positive_roots])

        elif basis.lower() == 'orthogonal':
            return positive_roots

        elif basis.lower() == 'alpha':
            return np.array([self._orthogonal_to_alpha(root) for root in
                             positive_roots])

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def _orthogonal_to_dynkin(self, vector):
        """Is a function that acts on a vector in the orthogonal basis.
        This function converts from the orthogonal to Dynkin basis.
        """
        return np.array([vector[i] - vector[i+1]
                        if i < (self.rank - 1) else 2*vector[i]
                        for i in range(self.rank)])

    def _orthogonal_to_alpha(self, vector):
        """Acts on a vector in the orthogonal basis. It converts it into
        a vector in the alpha basis.
        """
        new_vec = []
        new_val = 0
        for i in range(self.rank):
            new_val += vector[i]
            new_vec.append(new_val)
        return new_vec

    def _dynkin_to_orthogonal(self, vector):
        """Acts on a vector in the Dynkin basis.
        This converts from the Dynkin basis to the orthogonal basis.
        """
        new_vec = []
        new_val = np.sum(vector) - vector[self.rank - 1]/2
        for i in range(self.rank):
            new_vec.append(new_val)
            new_val -= vector[i]
        return new_vec

    def _dynkin_to_alpha(self, vector):
        """We simply convert the Dynkin into orthogonal, and then orthogonal
        into alpha basis.
        """
        return self._orthogonal_to_alpha(self._dynkin_to_orthogonal(vector))

    def _alpha_to_dynkin(self, vector):
        """Converts the alpha basis into the Dynkin basis. This is much easier
        with the transpose of the Cartan matrix.
        """
        return list(np.dot(self.cartan_matrix().T, vector))

    def _alpha_to_orthogonal(self, vector):
        """Alpha to Dynkin, then Dynkin to orthogonal.
        """
        return self._dynkin_to_orthogonal(self._alpha_to_dynkin(vector))

# C_N / SP(N) Groups


class CGroup(Group):
    """Data for the C_N or SP(N) representations. The rank needs to be a
    positive integer.
    """

    def __new__(cls, rank):
        if rank > 0 and type(rank) == int:
            return super(CGroup, cls).__new__(cls)

        else:
            raise Exception('The rank needs to be a positive integer')

    def __init__(self, rank):
        super().__init__(rank)

    def __repr__(self):
        return 'C' + str(self.rank)

    def _simple_roots(self, basis='dynkin'):
        """The matrix form of the simple roots for calculations."""
        simple_roots = np.zeros((self.rank, self.rank), dtype=int)
        for i in range(self.rank):

            if i == self.rank - 1:
                simple_roots[i, :] = 2 * orth_basis(self.rank)[i, :]
            else:
                simple_roots[i, :] = (orth_basis(self.rank)[i, :]
                                      - orth_basis(self.rank)[i + 1, :])

        if basis.lower() == 'dynkin':
            return np.array([self._orthogonal_to_dynkin(root) for root in
                             simple_roots])
        elif basis.lower() == 'orthogonal':
            return simple_roots

        elif basis.lower() == 'alpha':
            return np.array([self._orthogonal_to_alpha(root) for root in
                             simple_roots])

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def _positive_roots(self, basis='dynkin'):
        """The matrix form of the positive roots for calculations."""
        positive_roots = np.zeros(((self.rank)*(self.rank), self.rank),
                                  dtype=int)

        k = 0
        for i in range(0, self.rank):
            for j in range(i, self.rank):
                if j == i:
                    positive_roots[k, :] = 2 * orth_basis(self.rank)[i, :]
                    k += 1
                else:
                    positive_roots[k, :] = (orth_basis(self.rank)[i, :]
                                            + orth_basis(self.rank)[j, :])
                    k += 1
                    positive_roots[k, :] = (orth_basis(self.rank)[i, :]
                                            - orth_basis(self.rank)[j, :])
                    k += 1

        if basis.lower() == 'dynkin':
            return np.array([self._orthogonal_to_dynkin(root) for root in
                             positive_roots])

        elif basis.lower() == 'orthogonal':
            return positive_roots

        elif basis.lower() == 'alpha':
            return np.array([self._orthogonal_to_alpha(root) for root in
                             positive_roots])

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def _orthogonal_to_dynkin(self, vector):
        """Is a function that acts on a vector in the orthogonal basis.
        This function converts from the orthogonal to Dynkin basis.
        """
        return np.array([vector[i] - vector[i+1]
                        if i < (self.rank - 1) else vector[i]
                        for i in range(self.rank)])

    def _orthogonal_to_alpha(self, vector):
        """Acts on a vector in the orthogonal basis. It converts it into
        a vector in the alpha basis.
        """
        new_vec = []
        new_val = 0
        for i in range(self.rank):
            new_val += vector[i]
            if i < self.rank - 1:
                new_vec.append(new_val)
            else:
                new_vec.append(new_val/2)
        return new_vec

    def _dynkin_to_orthogonal(self, vector):
        new_vec = []
        new_val = np.sum(vector)
        new_vec.append(new_val)
        for i in range(self.rank - 1):
            new_val -= vector[i]
            new_vec.append(new_val)
        return new_vec

    def _dynkin_to_alpha(self, vector):
        """We simply convert the Dynkin into orthogonal, and then orthogonal
        into alpha basis.
        """
        return self._orthogonal_to_alpha(self._dynkin_to_orthogonal(vector))

    def _alpha_to_dynkin(self, vector):
        """Converts the alpha basis into the Dynkin basis. This is much easier
        with the transpose of the Cartan matrix.
        """
        return list(np.dot(self.cartan_matrix().T, vector))

    def _alpha_to_orthogonal(self, vector):
        """Alpha to Dynkin, then Dynkin to orthogonal.
        """
        return self._dynkin_to_orthogonal(self._alpha_to_dynkin(vector))

# D_N / SO(2N) Groups


class DGroup(Group):
    """Data for the D_N or SO(2N) representations. The rank needs to be a
    positive integer greater than 1.
    """
    def __new__(cls, rank):
        if rank > 1 and type(rank) == int:
            return super(DGroup, cls).__new__(cls)

        if rank == 1:
            raise Exception('D_N is defined only for N > 1')
        else:
            raise Exception('The rank needs to be a positive integer')

    def __init__(self, rank):
        super().__init__(rank)

    def __repr__(self):
        return 'D' + str(self.rank)

    def _simple_roots(self, basis='dynkin'):
        """The matrix form of the simple roots for calculations."""
        simple_roots = np.zeros((self.rank, self.rank), dtype=int)
        for i in range(self.rank):
            if i == self.rank - 1:
                simple_roots[i, :] = (orth_basis(self.rank)[i - 1, :]
                                      + orth_basis(self.rank)[i, :])
            else:
                simple_roots[i, :] = (orth_basis(self.rank)[i, :]
                                      - orth_basis(self.rank)[i + 1, :])

        if basis.lower() == 'dynkin':
            return np.array([self._orthogonal_to_dynkin(root) for root in
                             simple_roots])

        elif basis.lower() == 'orthogonal':
            return simple_roots

        elif basis.lower() == 'alpha':
            return np.array([self._orthogonal_to_alpha(root) for root in
                             simple_roots])

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def _positive_roots(self, basis='dynkin'):
        """The matrix form of the positive roots for calculations."""
        positive_roots = np.zeros(((self.rank)*(self.rank) - self.rank,
                                  self.rank), dtype=int)

        k = 0
        for i in range(0, self.rank - 1):
            for j in range(i + 1, self.rank):
                positive_roots[k, :] = (orth_basis(self.rank)[i, :]
                                        + orth_basis(self.rank)[j, :])
                k += 1
                positive_roots[k, :] = (orth_basis(self.rank)[i, :]
                                        - orth_basis(self.rank)[j, :])
                k += 1

        if basis.lower() == 'dynkin':
            return np.array([self._orthogonal_to_dynkin(root) for root in
                             positive_roots])

        elif basis.lower() == 'orthogonal':
            return positive_roots

        elif basis.lower() == 'alpha':
            return np.array([self._orthogonal_to_alpha(root) for root in
                             positive_roots])

        else:
            raise Exception('Please use the orthogonal basis or the dynkin \
            basis.')

    def _orthogonal_to_dynkin(self, vector):
        """Is a function that acts on a vector in the orthogonal basis.
        This function converts from the orthogonal to Dynkin basis.
        """
        return np.array([vector[i] - vector[i+1]
                        if i < (self.rank - 2)
                        else vector[self.rank - 2] - vector[self.rank - 1]
                        if i == self.rank - 2
                        else vector[self.rank - 2] + vector[self.rank - 1]
                        for i in range(self.rank)])

    def _orthogonal_to_alpha(self, vector):
        """Acts on a vector in the orthogonal basis. It converts it into
        a vector in the alpha basis.
        """
        new_vec = []
        new_val = 0
        for i in range(self.rank - 2):
            new_val += vector[i]
            new_vec.append(new_val)

        new_vec.append((new_val + vector[self.rank - 2]
                        - vector[self.rank - 1])/2)
        new_vec.append((new_val + vector[self.rank - 2]
                        + vector[self.rank - 1])/2)
        return new_vec

    def _dynkin_to_orthogonal(self, vector):
        """Acts on a vector in the Dynkin basis.
        This converts from the Dynkin basis to the orthogonal basis.
        """
        new_vec = []
        new_val = (np.sum(vector) - vector[self.rank - 2]/2
                   - vector[self.rank - 1]/2)

        new_vec.append(new_val)
        for i in range(self.rank - 1):
            new_val -= vector[i]
            new_vec.append(new_val)
        return new_vec

    def _dynkin_to_alpha(self, vector):
        """We simply convert the Dynkin into orthogonal, and then orthogonal
        into alpha basis.
        """
        return self._orthogonal_to_alpha(self._dynkin_to_orthogonal(vector))

    def _alpha_to_dynkin(self, vector):
        """Converts the alpha basis into the Dynkin basis. This is much easier
        with the transpose of the Cartan matrix.
        """
        return list(np.dot(self.cartan_matrix().T, vector))

    def _alpha_to_orthogonal(self, vector):
        """Alpha to Dynkin, then Dynkin to orthogonal.
        """
        return self._dynkin_to_orthogonal(self._alpha_to_dynkin(vector))

#################################################################
# Representations & Product Representations
#################################################################


class Representation:
    """This class allows us to use the group data to construct and explore
    representations. We work in the Dynkin basis.
    """

    def __init__(self, group, highest_weight):
        self.group = group
        self.highest_weight = highest_weight
        self.rank = len(self.highest_weight)

    def dim(self):
        denom = 1
        num = 1
        rho = self.group(self.rank).weyl_vector()
        hw = np.array(self.highest_weight)
        pos_roots = self.group(self.rank)._positive_roots()

        for root in pos_roots:
            denom *= self.group(self.rank).inner_product(rho, root)
            num *= self.group(self.rank).inner_product(hw + rho, root)
        return int(round(num/denom))

    def dominant_weights(self):
        """Return the set of dominant weights of a representation."""
        pos_roots = self.group(self.rank)._positive_roots()
        dom_weights = [tuple(self.highest_weight)]

        queue = deque()
        queue.append(tuple(self.highest_weight))

        # implementing a simple algorithm for finding dominant roots
        while queue:
            dom = np.array(queue.popleft())
            for root in pos_roots:
                trial_weight = dom - root
                trial_weight = tuple(trial_weight)

                # check if the root is dominant
                if np.all(np.array(trial_weight) >= 0):
                    if trial_weight not in queue:
                        queue.append(trial_weight)
                    dom_weights.append(trial_weight)

        dom_weights = [*map(list, dom_weights)]
        return dom_weights

    def weight_level(self, weight):
        """Given a weight of a representation, find its level relative to
        the highest weight.
        """
        group = self.group(self.rank)
        hw_alpha = np.array(group._dynkin_to_alpha(self.highest_weight))
        weight_alpha = np.array(group._dynkin_to_alpha(weight))
        return int(round(np.sum(hw_alpha - weight_alpha)))

    def weight_system(self, form='list'):
        """Can either have it as a list or array for manipulation, or a
        dictionary if you are interested in specific levels.

        :param form: can be 'list', 'dict' or 'array'

        :output: either a list, dictionary or array.

        >>> R = Representation(BGroup, [1,0,0])
        >>> R.weight_system(form='array')
        array([[ 1,  0,  0],
            [-1,  1,  0],
            [ 0, -1,  2],
            [ 0,  0,  0],
            [ 0,  1, -2],
            [ 1, -1,  0],
            [-1,  0,  0]])
        """

        if form == 'dict':
            return self._build_module(with_level=True)

        elif form == 'list':
            return self._build_module(with_level=False)

        elif form == 'array':
            return np.array(self._build_module(with_level=False))

    def print_weights(self):
        """Prints the weight system as a level decomposition or a spindle
        shape. If one wants a list form then just print the list returned by weight_system().

        >>> R = Representation(DGroup, [1,0,1])
        >>> R.print_weights()
        Level 0:  (1, 0, 1)
        Level 1:  (-1, 1, 2) (2, 0, -1)
        Level 2:  (0, 1, 0) (0, 1, 0) (0, -1, 2)
        Level 3:  (1, 1, -2) (-2, 2, 1) (1, -1, 0) (1, -1, 0)
        Level 4:  (-1, 2, -1) (-1, 0, 1) (-1, 0, 1) (2, -1, -2)
        Level 5:  (0, 0, -1) (0, 0, -1) (0, -2, 1)
        Level 6:  (1, -2, -1) (-2, 1, 0)
        Level 7:  (-1, -1, 0)
        """
        d = self._build_module(with_level=True)
        for key, value in d.items():
        return

    def _freudenthal_raw(self, weight, module, mult):
        """Implementation of the Freudenthal algorithm.
        Give multiplicities as a dictionary so that it can recursively
        fill out information.
        The module is the list of weights that we have already determined
        to be in the representation.
        Mult is a dictionary that we provide, which you can initiate
        to be empty as long as you start with the highest weight.
        """
        highest_weight = self.highest_weight

        if weight == highest_weight:
            mult_num = 1
            return mult_num

        rank = self.rank
        group = self.group(rank)
        positive_roots = group._positive_roots()
        rho = group.weyl_vector()
        # we use the recursive algorithm to increment the RHS
        RHS = 0
        for root in positive_roots:
            k = 1
            while tuple(np.array(weight) + k * root) in module:
                part_sum = (group.inner_product(np.array(weight)
                            + k * root, root)
                            * mult[tuple(np.array(weight) + k * root)]) * 2
                RHS += part_sum
                k += 1

        # now divide by the norm factor we defined
        norm = (group.inner_product(highest_weight + rho, highest_weight + rho)
                - group.inner_product(weight + rho, weight + rho))

        mult_num = round(RHS / norm)
        return int(mult_num)

    def _build_module(self, with_level=False):
        """Builds the module using the Dynkin label method.

        Note: This method is only efficient when the order of the dimension
        of the representation is of order rank * rank. For ranks above this,
        one should use Weyl orbit method.

        :var with_level: True or False, depending on whether you want
        to know the level of a state in a module.
        :return: returns a list of a list of states in a module. Returns
        a dictionary where the key is the level and the value is the
        sub-module at that level if with_level is True.

        >>> R = Representation(AGroup, [1,0,0])
        >>> R._build_module(with_level=True)
        {'Level 0': [(1, 0, 0)], 'Level 1': [(-1, 1, 0)],
        'Level 2': [(0, -1, 1)], 'Level 3': [(0, 0, -1)]}
        >>> R._build_module()
        [(1, 0, 0), (-1, 1, 0), (0, -1, 1), (0, 0, -1)]
        """
        highest_weight = np.array(self.highest_weight)
        group = self.group(self.rank)
        simple_roots = group._simple_roots()

        if np.any(highest_weight < 0):
            print('Enter Dynkin labels greater than or equal to zero.')
            return

        # the Freudenthal multiplicity formula requires that all states above
        # the one we are considering need to be fleshed out. For this reason,
        # we have to first generate the entire module, then deal with the
        # multiplicites on a second pass through, which we do level by level.

        weight_set = set()
        weight_set.add(tuple(highest_weight))

        # we instantiate a dictionary whose keys run over the allowed levels
        hw_alpha = group._dynkin_to_alpha(self.highest_weight)
        max_height = int(round(2 * np.sum(hw_alpha)))
        module = {f'Level {x}': set() for x in range(max_height + 1)}
        module['Level 0'].add(tuple(highest_weight))

        # now we flesh out the module using the Dynkin number approach
        queue = deque()
        queue.append([highest_weight, 0])
        while queue:
            w = queue.pop()
            for item, dynk in enumerate(w[0]):
                weight = w[0]
                level = w[1]
                if dynk > 0:
                    for _ in range(dynk):
                        weight = weight - simple_roots[item]
                        level += 1

                        queue.append([weight, level])
                        weight = tuple(weight)
                        module[f'Level {level}'].add(weight)
                        weight_set.add(weight)

        mult = {}
        mult[tuple(highest_weight)] = self._freudenthal_raw(
            list(highest_weight), weight_set, mult
            )

        if with_level:
            for level in range(0, len(module)):
                new_level = []
                for state in module[f'Level {level}']:
                    mult[state] = int(
                        self._freudenthal_raw(list(state), weight_set, mult)
                        )
                    new_level += [state] * mult[state]
                module[f'Level {level}'] = new_level
            return module

        else:
            new_module = []
            for level in range(0, len(module)):
                for state in module[f'Level {level}']:
                    mult[state] = int(
                        self._freudenthal_raw(list(state), weight_set, mult)
                        )
                    new_module += [state] * mult[state]
            return new_module


class ProductRep:
    """This class allows us to use the group data and standard representation
    data to form composite representations of product symmetry groups.

    For example, we can use it to represent a state in a representation of
    SU(2) x SU(2) x SU(4).

    We would initialise this with:
    ProductRep((AGroup, [1]), (AGroup, [1]), (AGroup, [1, 0, 0]))
    """

    def __init__(self, *args):
        self.args = args
        self.unranked_groups = [group[0] for group in self.args]
        self.groups = [group(len(weight)) for group, weight in self.args]
        self.ranks = [len(weight) for group, weight in self.args]
        self.weights = [weight for group, weight in self.args]
        self.vector = [num for weight in self.weights for num in weight]

    def __repr__(self):
        return str([f'{group}{weight}' for group, weight
                    in zip(self.groups, self.weights)])

    def _to_vec(self, vector):
        """Takes a vector of the form self.vector and converts it into the form
        self.weights.
        """
        return [[vector.pop(0) for _ in range(rank)]
                for rank in self.ranks]

    def _rep_concat(self, weights, group_and_rep):
        """Take a list of weights and an input (group, highest_weight) and
        concatenate the two with the weight system generated by the latter
        argument.

        :param weights: list of weights generated from a representation
        :param group_and_rep: A tuple of group and highest weight

        :output: a list of concatenated weight system vectors.

        >>> pr = ProductRep((AGroup, [1, 0]), (BGroup, [0, 1]))
        >>> rep = Representation(AGroup, [1,0,0])
        >>> weights = rep.weight_system()
        >>> pr._rep_concat(weights, (BGroup, [0, 1]))
        [(1, 0, 0, 0, 1), (1, 0, 0, 1, -1), (1, 0, 0, -1, 1), (1, 0, 0, 0, -1),
        (-1, 1, 0, 0, 1), (-1, 1, 0, 1, -1), (-1, 1, 0, -1, 1), (-1, 1, 0, 0,
        -1), (0, -1, 1, 0, 1), (0, -1, 1, 1, -1), (0, -1, 1, -1, 1), (0, -1, 1,
        0, -1), (0, 0, -1, 0, 1), (0, 0, -1, 1, -1), (0, 0, -1, -1, 1), (0, 0,
        -1, 0, -1)]
        """
        rep = Representation(group_and_rep[0], group_and_rep[1])
        new_weights = rep.weight_system()
        return [x + y for x in weights for y in new_weights]

    def build(self, form='array'):
        """Builds the composite representation by applying the representation
        building techniques for each constituent part.

        :param form: form is either 'array' or 'list'

        :output: either an array or a list

        >>> PR = ProductRep((CGroup, [0, 1]), (BGroup, [0, 1]))
        >>> PR.build()
        array([[ 0,  1,  0,  1],
           [ 0,  1,  1, -1],
           [ 0,  1, -1,  1],
           [ 0,  1,  0, -1],
           [ 2, -1,  0,  1],
           [ 2, -1,  1, -1],
           [ 2, -1, -1,  1],
           [ 2, -1,  0, -1],
           [ 0,  0,  0,  1],
           [ 0,  0,  1, -1],
           [ 0,  0, -1,  1],
           [ 0,  0,  0, -1],
           [-2,  1,  0,  1],
           [-2,  1,  1, -1],
           [-2,  1, -1,  1],
           [-2,  1,  0, -1],
           [ 0, -1,  0,  1],
           [ 0, -1,  1, -1],
           [ 0, -1, -1,  1],
           [ 0, -1,  0, -1]])

        >>> PR = ProductRep((AGroup, [0, 1]), (BGroup, [0, 1]))
        >>> PR.build(form='list')
        [(0, 1, 0, 1), (0, 1, 1, -1), (0, 1, -1, 1), (0, 1, 0, -1), (1, -1, 0,
        1), (1, -1, 1, -1), (1, -1, -1, 1), (1, -1, 0, -1), (-1, 0, 0, 1), (-1,
        0, 1, -1), (-1, 0, -1, 1), (-1, 0, 0, -1)]
        """
        tmp_list = list(self.args)
        # we need to have the first element as a list in order to reduce
        tmp_list[0] = Representation(tmp_list[0][0], tmp_list[0][1])
        tmp_list[0] = tmp_list[0].weight_system()

        if form == 'array':
            return np.array(reduce(self._rep_concat, tmp_list))

        elif form == 'list':
            return reduce(self._rep_concat, tmp_list)

        else:
            raise Exception('form needs to be list or array')

    def product_racah(self, vector):
        """Produces a racah-speiser reflection of the composite
        representation. Note that it will only reflect with respect to the
        groups that are specified in the ProductRep object. Make sure you
        correctly order the Dynkin weights in your product vector.

        :param vector: The vector of dynkin labels that you want to reflect.

        :output: A vector of the form [new_vector, number of bounces]

        >>> pr = ProductRep((AGroup, [1, 0, 0]), (BGroup, [0, 1]))
        >>> pr.product_racah([-2, 3, 0, 1, 1])
        [[0, 2, 0, 1, 1], 1]

        >>> pr = ProductRep((AGroup, [1, 0]), (CGroup, [0, 1]))
        >>> pr.product_racah([1, 0, -2, 0])
        [[1, 0, -2, 0], 'del']
        """
        if len(vector) == np.sum(self.ranks):
            sub_vecs = self._to_vec(vector)
            reps = zip(self.unranked_groups, sub_vecs)
            new_vec = []
            bounces = 0

            for rep in reps:
                lt = ListTensor(rep[0], [rep[1]])
                bounced_vec = lt.racah_reflection(rep[1])

                new_vec += list(bounced_vec[0])
                if bounced_vec[1] == 'del' or bounces == 'del':
                    bounces = 'del'
                else:
                    bounces += bounced_vec[1]
            return [new_vec, bounces]
        else:
            raise Exception('Vector is not the same rank.')


#################################################################
# Tensor products
#################################################################


class ListTensor:
    """This class allows us to use the take tensor products of a list of
    representations of the algebras. We initialise with a group and a list of
    highest weights of representations.

    It is advised to be the highest dimension representations first in the
    list.
    """
    def __new__(cls, group, highest_weight_list):
        rank = len(highest_weight_list[0])
        valid = True
        for weight in highest_weight_list:
            if len(weight) != rank:
                valid = False
                break

        if valid:
            return super(ListTensor, cls).__new__(cls)
        else:
            raise Exception('The representations should be all same rank')

    def __init__(self, group, highest_weight_list):
        self.group = group
        self.weights = highest_weight_list
        self.rank = len(self.weights[0])

    def _racah_bounce(self, vec_and_bounces):
        """Input is a vector of the for [vector, number of bounces] to keep
        track of the number of bounces. It returns a list of bounced vectors
        along with the number of times they have been bounced.
        """
        vec = vec_and_bounces[0]
        num = vec_and_bounces[1]
        bounced_vecs = []

        for func in self.group(self.rank).racah_reflections():
            bounced_vecs.append([np.round(func(vec)).astype('int'), num + 1])
        return bounced_vecs

    def racah_reflection(self, arr, no_of_reflect=0):
        """Performs as many reflections as needed to obtain a weight in the
        dominant Weyl chamber. It also keeps track of the number bounces that
        were used.
        """
        queue = deque()
        queue.append([arr, no_of_reflect])
        state = queue.popleft()

        while min(state[0]) < 0:
            for vec_and_bounce in self._racah_bounce(state):
                queue.append(vec_and_bounce)

            state = queue.popleft()
            # check that the state is not equal to itself up to a minus sign
            if list(state[0]) == list(arr) and state[1] % 2 == 1:
                return [state[0], 'del']
        return state

    def _two_state_racah(self, weight1, weight2):
        """Computes the Racah Speiser algorithm on two representations.
        Note that it takes highest weight states as input. The output is a
        dictionary of highest weight representations and their multiplicities.
        """

        if len(weight1) != len(weight2):
            print('representations not from the same group.')
            return

        rep1 = Representation(self.group, list(weight1))
        dim1 = rep1.dim()
        rep2 = Representation(self.group, list(weight2))
        dim2 = rep2.dim()

        states = {}

        if dim1 >= dim2:
            hw = np.array(rep1.highest_weight)
            all_weights2 = rep2.weight_system(form='array')
            for weight in all_weights2:
                new_state = self.racah_reflection(weight + hw)

                if new_state[1] != 'del':
                    if new_state[1] % 2 == 0:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) + 1
                    elif new_state[1] % 2 == 1:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) - 1

        else:
            hw = np.array(rep2.highest_weight)
            all_weights1 = rep1.weight_system(form='array')
            for weight in all_weights1:
                new_state = self.racah_reflection(weight + hw)

                if new_state[1] != 'del':
                    if new_state[1] % 2 == 0:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) + 1
                    elif new_state[1] % 2 == 1:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) - 1

        return {key: value for key, value in states.items() if value != 0}

    def decompose(self, form='dict'):
        """Decomposes the multiple tensor product provided into a dictionary
        of highest weights and multiplicities.

        :param form: can be 'list', 'dict' or 'print'

        :output: either a list, dictionary or printed (with no output)
        """
        # initialise the states dictionary with the first two states
        weight_list = self.weights[:]  # create a copy to not erase attribute
        states_dict = self._two_state_racah(weight_list.pop(0),
                                            weight_list.pop(0))

        while weight_list:
            new_dict = {}
            weight = weight_list.pop(0)
            for state, mult in states_dict.items():
                temp_dict = self._two_state_racah(state, weight)
                for new_state, new_mult in temp_dict.items():
                    new_dict[new_state] = (new_dict.get(new_state, 0)
                                           + new_mult * mult)
            states_dict = new_dict

        if form == 'dict':
            return states_dict

        elif form == 'list':
            states = []
            for state, mult in states_dict.items():
                states += [state] * mult
            return states

        elif form == 'print':
            i = 1
            for state, mult in states_dict.items():
                if mult == 1:
                    print(f'{state}', end=' ')
                else:
                    print(f'{mult}x{state}', end=' ')
                if i < len(states_dict):
                    print(u"\u2295", end=' ')
                    i += 1
            print('')
            return

        else:
            print('form must be \'dict\', \'list\' or \'print\'')
            return


class RepTensor:
    """This class allows us to use the take tensor products of two
    representation-class objects.
    """
    def __new__(cls, repr1, repr2):
        if (repr1.group == repr2.group) and (repr1.rank == repr2.rank):
            return super(RepTensor, cls).__new__(cls)
        else:
            raise Exception('The groups should be the same type and same rank')

    def __init__(self, repr1, repr2):
        self.group = repr1.group
        self.rank = repr1.rank
        self.repr1 = repr1
        self.repr2 = repr2
        self.weight1 = repr1.highest_weight
        self.weight2 = repr2.highest_weight

    def _racah_bounce(self, vec_and_bounces):
        """Input is a vector of the for [vector, number of bounces] to keep
        track of the number of bounces. It returns a list of bounced vectors
        along with the number of times they have been bounced.
        """
        vec = vec_and_bounces[0]
        num = vec_and_bounces[1]
        bounced_vecs = []

        for func in self.group(self.rank).racah_reflections():
            bounced_vecs.append([np.round(func(vec)).astype('int'), num + 1])
        return bounced_vecs

    def racah_reflection(self, arr, no_of_reflect=0):
        """Performs as many reflections as needed to obtain a weight in the
        dominant Weyl chamber. It also keeps track of the number bounces that
        were used.
        """
        queue = deque()
        queue.append([arr, no_of_reflect])
        state = queue.popleft()

        while min(state[0]) < 0:
            for vec_and_bounce in self._racah_bounce(state):
                queue.append(vec_and_bounce)

            state = queue.popleft()
            # check that the state is not equal to itself up to a minus sign
            if list(state[0]) == list(arr) and state[1] % 2 == 1:
                return [state[0], 'del']
        return state

    def decompose(self, form='dict'):
        """Computes the Racah Speiser algorithm on the two representations
        given as we initialised the object.

        :param form: can be 'list', 'dict' or 'print'

        :output: either a list, dictionary or printed (with no output)
        """
        states = {}
        dim1 = self.repr1.dim()
        dim2 = self.repr2.dim()

        if dim1 >= dim2:
            hw = np.array(self.weight1)
            all_weights2 = self.repr2.weight_system(form='array')
            for weight in all_weights2:
                new_state = self.racah_reflection(weight + hw)

                if new_state[1] != 'del':
                    if new_state[1] % 2 == 0:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) + 1
                    elif new_state[1] % 2 == 1:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) - 1

        else:
            hw = np.array(self.weight2)
            all_weights1 = self.repr1.weight_system(form='array')
            for weight in all_weights1:
                new_state = self.racah_reflection(weight + hw)

                if new_state[1] != 'del':
                    if new_state[1] % 2 == 0:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) + 1
                    elif new_state[1] % 2 == 1:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) - 1

        weights = {key: value for key, value in states.items() if value != 0}

        if form == 'dict':
            return weights

        elif form == 'list':
            states = []
            for state, mult in weights.items():
                states += [state] * mult
            return states

        elif form == 'print':
            i = 1
            for state, mult in weights.items():
                if mult == 1:
                    print(f'{state}', end=' ')
                else:
                    print(f'{mult}x{state}', end=' ')
                if i < len(weights):
                    print(u"\u2295", end=' ')
                    i += 1
            print('')
            return

        else:
            print('form must be \'dict\', \'list\' or \'print\'')
            return


#################################################################
# SUSY module
#################################################################

class SUSYModule:
    """Initialise the supersymmetry module with the highest weight state, and
    then as many ProductRep objects as needed for your supersymmetry
    generators. In most cases this is just one, but in 4d where SO(4) = SU(2) x
    SU(2), the SUSY generators come in two distinct representations.

    Two examples are for 5d N=1:
    q = ProductRep((AGroup, [1]), (BGroup, [1, 0]))
    module_5d = SUSYModule([1, 0, 0, 0], q)

    For 4d N=2:
    q = ProductRep((AGroup, [1]), (AGroup, [0]), (AGroup, [1]))
    q_bar = ProductRep((AGroup, [0]), (AGroup, [1]), (AGroup, [1]))
    module_4d = SUSYModule([1, 0, 0], q, q_bar)

    Specifically one needs to initialise with ProductRep objects, which will
    be the representations of the supersymmetry generators in the maximally
    compact subalgebra.
    """
    def __new__(cls, highest_weight, *prodreps):
        for prodrep in prodreps:
            if np.sum(prodrep.ranks) != len(highest_weight):
                raise Exception('The highest weight should be the same rank as\
                    the algebra.')
            else:
                return super(SUSYModule, cls).__new__(cls)

    def __init__(self, highest_weight, *prodreps):
        self.prodreps = prodreps
        self.susy_vecs = [vector for prodrep in self.prodreps for vector in
                          prodrep.build(form='list')]
        self.constraints = {}
        self.groups = self.prodreps[0].groups
        self.ranks = self.prodreps[0].ranks
        self.heighest_weight = np.array(highest_weight)
        self.max_level = 0

    def get_susy_generators(self):
        """Returns the vectors of the supersymmetry generators.

        :output: list of Dynkin vectors for the SUSY representation.

        >>> pr = ProductRep((BGroup, [0, 1]), (AGroup, [1]))
        >>> sm = SUSYModule([-2, 0, 1], pr)
        >>> sm.get_susy_generators()
        [(0, 1, 1), (0, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1),
        (-1, 1, -1), (0, -1, 1), (0, -1, -1)]
        """
        return self.susy_vecs

    def add_constraints(self, level, removed_states):
        """One needs to add the constraints level by level in order to
        correctly reproduce the multiplet. Level 1 constraints are handled
        differently to Levels 2-4. This is because Level 1 constraints simply
        remove the supercharge from the basis entirely. The removed_states
        should be a list the quantum numbers of the combined set of
        supercharges that you wish to constrain.

        The constraint vectors need to be tuples because lists are not
        hashable, therefore we cannot use the set data structure which we later
        rely on.
        """
        self.constraints[f'Level {level}'] = self.constraints.get(
                f'Level {level}', []) + removed_states

    def _multiplicity_collapse(self, weights):
        """Takes a list of vectors, reflects them and then returns a
        dictionary of weights and multiplicities.
        """
        states = {}
        reflect = self.prodreps[0].product_racah
        for weight in weights:
                new_state = reflect(weight)

                if new_state[1] != 'del':
                    if new_state[1] % 2 == 0:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) + 1
                    elif new_state[1] % 2 == 1:
                        states[tuple(new_state[0])] = states.get(
                            tuple(new_state[0]), 0) - 1
        return states

    def _is_constrained(self, vector, constraints):
        """This is a helper function to determine whether, given a list of
        constraints (the combination of supercharges that result in a null
        state), the state in question should be removed form the verma module.
        """
        for constraint in constraints:
            # if the constraint is a subset of the vector then the vector is
            # constrained
            if set(vector) >= constraint:
                return True
            else:
                return False

    def build_module(self):
        """Constructs the SUSY module based on the information given. Note that
        we return a generator since the modules will be very large. The
        generator iterates over the levels of the SUSY module.
        """

        hw = self.heighest_weight
        supercharges = self.susy_vecs
        # unconstrained moves on as usual
        if not self.constraints:
            for level in range(len(supercharges) + 1):
                if level == 0:
                    yield {tuple(hw): 1}
                else:
                    states = [list(hw + np.sum(Q, axis=0)) for Q
                              in combinations(supercharges, level)]
                    states = self._multiplicity_collapse(states)

                    states = {key: value for key, value in states.items()
                              if value != 0}
                    yield states

        # if the module has level one constraints then life is made easier.
        elif 'Level 1' in self.constraints:
            # we just remove the level 1 states from the set of Qs
            for vec in self.constraints['Level 1']:
                    supercharges.remove(vec)
            # there still might be more constraints so we still need to split.
            if len(self.constraints) == 1:
                # if it is length one then there are no other constraints.
                for level in range(len(supercharges) + 1):
                    if level == 0:
                        yield {tuple(hw): 1}
                    else:
                        states = [list(hw + np.sum(Q, axis=0)) for Q
                                  in combinations(supercharges, level)]
                        states = self._multiplicity_collapse(states)

                        states = {key: value for key, value in states.items()
                                  if value != 0}
                        yield states
            else:
                # first we concatenate all higher level constraints into one
                # list. We do this first so we don't repeat it.
                higher_constraints = [set(constr) for level, constr_list in
                                      self.constraints.items() if level !=
                                      'Level 1' for constr in constr_list]

                for level in range(len(supercharges) + 1):
                    if level == 0:
                        yield {tuple(hw): 1}
                    else:
                        # an additional step is needed to check if the state
                        # we construct is constrained.
                        states = [list(hw + np.sum(Q, axis=0)) for Q
                                  in combinations(supercharges, level) if not
                                  self._is_constrained(Q, higher_constraints)]
                        states = self._multiplicity_collapse(states)

                        states = {key: value for key, value in states.items()
                                  if value != 0}
                        yield states
        # if there are no level 1 constraints then we don't remove Qs from the
        # basis at the start
        else:
            higher_constraints = [set(constr) for level, constr_list in
                                  self.constraints.items() for constr in
                                  constr_list]

            for level in range(len(supercharges) + 1):
                    if level == 0:
                        yield {tuple(hw): 1}
                    else:
                        states = [list(hw + np.sum(Q, axis=0)) for Q in
                                  combinations(supercharges, level) if not
                                  self._is_constrained(Q, higher_constraints)]
                        states = self._multiplicity_collapse(states)

                        states = {key: value for key, value in states.items()
                                  if value != 0}
                        yield states

    def return_module(self, form="print"):
        """Uses the iterator previously constructed in order return the states
        in an easier-to-read format.

        :param form: can be 'dict' or 'print'
        """
        module = self.build_module()
        output = {}
        # Keep track of the max level so we don't return a bunch of non
        # important levels (as opposed to there being an empty level due to an
        # EoM that might follow it)
        max_level = 0
        for level, states in enumerate(module):
            if len(states) != 0:
                output[f'Level {level}'] = states
                max_level = level

        for level in range(1, max_level):
            if f'Level {level}' not in output:
                output[f'Level {level}'] = {}

        if form == 'dict':
            return output

        else:
            for level, states in output.items():
                print('{:>8}:'.format(level), end=' ')
                i = 0
                for state, mult in states.items():
                    if i == 0:
                        if mult == 1:
                            print(f'{state}', end=' ')
                        elif mult > 1:
                            print(f'{mult}x{state}', end=' ')
                        elif mult == -1:
                            print(u"\u2296", end=' ')
                            print(f'{state}', end=' ')
                        else:
                            print(u"\u2296", end=' ')
                            print(f'{abs(mult)}x{state}', end=' ')
                        i += 1
                        if i == len(states):
                            print('')
                    else:
                        if mult == 1:
                            print(u"\u2295", end=' ')
                            print(f'{state}', end=' ')
                        elif mult > 1:
                            print(u"\u2295", end=' ')
                            print(f'{mult}x{state}', end=' ')
                        elif mult == -1:
                            print(u"\u2296", end=' ')
                            print(f'{state}', end=' ')
                        else:
                            print(u"\u2296", end=' ')
                            print(f'{abs(mult)}x{state}', end=' ')
                        i += 1
                        if i == len(states):
                            print('')


if __name__ == "__main__":
    flags = doctest.NORMALIZE_WHITESPACE
    doctest.testmod(optionflags=flags)
