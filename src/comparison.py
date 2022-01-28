import numpy as np
import scipy.sparse as sp
import sklearn.cluster as sl
from sklearn.preprocessing import normalize


def sqrtinvdiag(M):
    """Inverts and square-roots a positive diagonal matrix.
    Args:
        M (csc matrix): matrix to invert
    Returns:
        scipy sparse matrix of inverted square-root of diagonal
    """

    d = M.diagonal()
    dd = [1 / max(np.sqrt(x), 1 / 999999999) for x in d]

    return sp.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()


class Cluster:
    """Class containing all comparison clustering algorithms for directed networks.
        This should be initialised with a tuple of one csc matrix, representing the adjacency
        matrix. It contains clustering algorithms as methods and graph specifications
        as attributes.
        Args:
                A : adjacency matrix.
        num_clusters : the total number of clusters
        Attributes:
                A (csc matrix): adjacency matrix.
                D_in (csc matrix): diagonal in-degree matrix of adjacency.
                D_out (csc matrix): diagonal out-degree matrix of adjacency.
                Dbar (csc matrix): diagonal degree matrix.
                normA (csc matrix): symmetrically normalised adjacency matrix.
                size (int): number of nodes in network
        num_clusters : the total number of clusters
        H : Hermitian matrix from adjacency A
        H_rw : random walk normalized H
        """

    def __init__(self, A, num_clusters):
        self.A = A.tocsc()
        self.D_out = sp.diags(self.A.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        self.D_in = sp.diags(self.A.sum(axis=0).tolist(), [0]).tocsc()
        self.Dbar = (self.D_in + self.D_out)
        d = sqrtinvdiag(self.Dbar)
        self.normA = d * self.A * d
        self.size = self.A.shape[0]
        self.num_clusters = num_clusters
        self.H = (self.A-self.A.transpose()) * 1j
        # (np.real(self.H).power(2) + np.imag(self.H).power(2)).power(0.5)
        H_abs = np.abs(self.H)
        D_abs_inv = sp.diags(1/np.array(H_abs.sum(1))[:, 0])
        D_abs_inv.data[np.isinf(D_abs_inv.data)] = 0.0
        self.H_rw = D_abs_inv.dot(self.H)
        # below are for co-clustering
        tau = self.A.count_nonzero()/self.A.shape[0]
        P = np.power(np.array((self.A.sum(axis=0)+tau)), -0.5)  # column sums
        Q = np.power(np.array((self.A.sum(axis=1)+tau)), -0.5)  # row sums
        DISG_A = self.A.multiply(P).multiply(Q)
        self.DISG_U, _, self.DISG_Vt = sp.linalg.svds(
            DISG_A, k=self.num_clusters)

    def spectral_cluster_herm(self):
        u, s, vt = sp.linalg.svds(self.H, k=self.num_clusters)
        features_SVD = np.concatenate((np.real(u), np.imag(u)), axis=1)
        x = sl.KMeans(n_clusters=self.num_clusters).fit(features_SVD)
        return x.labels_

    def spectral_cluster_herm_rw(self):
        u, s, vt = sp.linalg.svds(self.H_rw, k=self.num_clusters)
        features_SVD = np.concatenate((np.real(u), np.imag(u)), axis=1)
        x = sl.KMeans(n_clusters=self.num_clusters).fit(features_SVD)
        return x.labels_

    def spectral_cluster_Bi_sym(self):
        U = self.A.dot(self.A.transpose()) + self.A.transpose().dot(self.A)
        U = U.asfptype()
        (w, v) = sp.linalg.eigsh(
            U, self.num_clusters, which='LM')  # largest magnitude
        x = sl.KMeans(n_clusters=self.num_clusters).fit(v)
        return x.labels_

    def spectral_cluster_DD_sym(self):
        D_in_invhalf = self.D_in.power(-0.5)
        D_out_invhalf = self.D_out.power(-0.5)
        U = D_out_invhalf.dot(self.A).dot(D_in_invhalf).dot(self.A.transpose()).dot(D_out_invhalf) + \
            D_in_invhalf.dot(self.A.transpose()).dot(
                D_out_invhalf).dot(self.A).dot(D_in_invhalf)
        U = U.asfptype()
        (w, v) = sp.linalg.eigsh(
            U, self.num_clusters, which='LM')  # largest magnitude
        x = sl.KMeans(n_clusters=self.num_clusters).fit(v)
        return x.labels_

    def spectral_cluster_DISG_L(self):
        x = sl.KMeans(n_clusters=self.num_clusters).fit(self.DISG_U)
        return x.labels_

    def spectral_cluster_DISG_R(self):
        x = sl.KMeans(n_clusters=self.num_clusters).fit(
            np.transpose(self.DISG_Vt))
        return x.labels_

    def spectral_cluster_DISG_LR(self):
        features = np.concatenate(
            (self.DISG_U, np.transpose(self.DISG_Vt)), axis=1)
        x = sl.KMeans(n_clusters=self.num_clusters).fit(features)
        return x.labels_

    def spectral_cluster_Laplacian(self):  # failure attempt...
        P = normalize(self.A, 'l1')
        P = P.asfptype()
        (evals, evecs) = sp.linalg.eigsh(P.transpose())
        evec1 = evecs[:, np.isclose(evals, 1, atol=0.1)]

        # Since np.isclose will return an array, we've indexed with an array
        # so we still have our 2nd axis.  Get rid of it, since it's only size 1.
        evec1 = evec1[:, 0]

        stationary = evec1 / evec1.sum()

        # eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
        stationary = stationary.real
        Pi = sp.diags(np.array(stationary))
        tmp = Pi.power(0.5).dot(P).dot(Pi.power(-0.5) +
                                       Pi.power(-0.5).dot(P.transpose())).dot(Pi.power(0.5))
        L = sp.eye(P.shape[0]) - 0.5*tmp
        (w, v) = sp.linalg.eigsh(L, self.num_clusters, which='SA')
        X = v[:, 1:]
        X_star = Pi.power(-0.5).dot(X)
        X_star = normalize(X_star, norm='l1', axis=0)
        x = sl.KMeans(n_clusters=self.num_clusters).fit(X_star)
        return x.labels_
