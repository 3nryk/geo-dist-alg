''' Code based from :
    https://gallantlab.github.io/auto_examples/surface_analyses/plot_geodesic_distance.html?highlight=distance
    https://github.com/gallantlab/pycortex/blob/master/cortex/polyutils/surface.py '''

import numpy as np
import numexpr as ne
from scipy import sparse
from scipy.sparse.linalg.dsolve import factorized

import matplotlib as mpl
import matplotlib.cm as cm

def vertex_face_matrix(pts, polys):
    """
    Sparse matrix of vertex-face associations
    :param pts: arrays of vertices
    :param polys: arrays of faces
    :return: sparse matrix with shape (num_pt, num_poly)
    """

    num_pt = len(pts)
    num_poly = len(polys)

    data = np.ones(3 * num_poly)
    row = np.hstack(polys.T)
    col = np.tile(range(num_poly), (1, 3)).squeeze()
    shp = (num_pt, num_poly)

    # coo_matrix()  := coordinate format matrix
    # tocsr()       := compressed sparse row format
    sm = sparse.coo_matrix((data, (row, col)), shp).tocsr()

    return sm


def face_areas(pts, polys):
    """
    Computes area of each face
    :param pts: arrays of vertices
    :param polys: arrays of faces
    :return: an array of areas (num_poly, )
    """

    # 3D matrix of points in each face: n faces x 3 points per face x 3 coordinates per point
    ppts = pts[polys]

    # compute normal vector
    norm_vec = np.cross(ppts[:, 1] - ppts[:, 0],
                        ppts[:, 2] - ppts[:, 0])

    # compute area (sum(-1) for sum in axis = -1)
    area_face = 0.5 * np.sqrt(np.square(norm_vec).sum(-1))

    return area_face


def cotangent_angles(pts, polys):
    """
    Computes cotangent of each angle of each face
    :param pts: arrays of vertices
    :param polys: arrays of faces
    :return: cotangents of each angle of each face (3, num_pts)
    """

    # 3D matrix of points in each face: n faces x 3 points per face x 3 coordinates per point
    ppts = pts[polys]

    # compute cotangents of each angle of each face
    cots1 = ((ppts[:, 1] - ppts[:, 0]) *
             (ppts[:, 2] - ppts[:, 0])).sum(1) / np.sqrt((np.cross(ppts[:, 1] - ppts[:, 0],
                                                                   ppts[:, 2] - ppts[:, 0]) ** 2).sum(1))
    cots2 = ((ppts[:, 2] - ppts[:, 1]) *
             (ppts[:, 0] - ppts[:, 1])).sum(1) / np.sqrt((np.cross(ppts[:, 2] - ppts[:, 1],
                                                                   ppts[:, 0] - ppts[:, 1]) ** 2).sum(1))
    cots3 = ((ppts[:, 0] - ppts[:, 2]) *
             (ppts[:, 1] - ppts[:, 2])).sum(1) / np.sqrt((np.cross(ppts[:, 0] - ppts[:, 2],
                                                                   ppts[:, 1] - ppts[:, 2]) ** 2).sum(1))

    # sanitize
    cots = np.vstack([cots1, cots2, cots3])
    cots[np.isinf(cots)] = 0
    cots[np.isnan(cots)] = 0

    return cots


def laplace_operator(pts, polys):
    """
    Laplace Beltrami operator for triangles.
    We first compute the diagonal mass matrix M which are the vertex areas
        vertex areas are 1/3 of the total areas of all the triangles incident to the vertex
    Then we compute the cotangent laplacian matrix C
    :param pts: arrays of vertices
    :param polys: arrays of faces
    :return: diagonal mass matrix M and cotangent laplacian matrix C
    """

    num_pt = len(pts)
    # Step 1: create diagonal mass matrix M (vertex areas A)
    vertex_areas = vertex_face_matrix(pts, polys).dot(face_areas(pts, polys)) / 3.0

    M = sparse.dia_matrix((vertex_areas, [0]), (num_pt, num_pt)).tocsr()

    # Step 2: create cotangent laplacian matrix C
    ## first, create W, weighted adjacency matrix
    # cots1, cots2, cots3 are the cotangent angles of each face
    cots1, cots2, cots3 = cotangent_angles(pts, polys)

    # suppose i and j are vertices, (ij is an edge)
    # get the values of Cij which is 0.5 * (cot(alpha_ij) + cot(beta_ij))
    # this is for alpha
    W1 = sparse.coo_matrix((cots1, (polys[:, 1], polys[:, 2])), (num_pt, num_pt))
    W2 = sparse.coo_matrix((cots2, (polys[:, 2], polys[:, 0])), (num_pt, num_pt))
    W3 = sparse.coo_matrix((cots3, (polys[:, 0], polys[:, 1])), (num_pt, num_pt))

    # for adding beta and dividing by 2
    # note that beta is the angle opposite of alpha given edge ij
    W = 0.5 * (W1 + W1.T + W2 + W2.T + W3 + W3.T).tocsr()

    # for the values of Cii = - sum (Cij) for all j neighbors of i
    # V is the sum
    V = sparse.dia_matrix((np.array(W.sum(0)).ravel(), [0]), (num_pt, num_pt))

    C = W - V

    return C, M


def avg_edge_length(pts, polys):
    """
    Computes average edge length, h, of all edges in the surface
    :param pts: arrays of vertices
    :param polys: arrays of faces
    :return: average edge length
    """

    # create a sparse vertex adjacency matrix
    num_pt = len(pts)
    num_poly = len(polys)

    # for each edge in the face
    adj1 = sparse.coo_matrix((np.ones((num_poly,)),
                              (polys[:, 0], polys[:, 1])), (num_pt, num_pt))
    adj2 = sparse.coo_matrix((np.ones((num_poly,)),
                              (polys[:, 0], polys[:, 2])), (num_pt, num_pt))
    adj3 = sparse.coo_matrix((np.ones((num_poly,)),
                              (polys[:, 1], polys[:, 2])), (num_pt, num_pt))
    alladj = (adj1 + adj2 + adj3).tocsr()

    tadj = sparse.triu(alladj + alladj.T, 1)
    edge_len = np.sqrt(np.square(pts[tadj.row] - pts[tadj.col]).sum(1))

    return edge_len.mean()

def surface_gradient(pts, polys, heat_values):
    """
    Computes gradient of the surface where each vertex of the surface has a heat value
    :param pts: arrays of vertices
    :param polys: arrays of faces
    :param heat_values: 1D array of heat values of all vertices
    :return: contains the x, y, z axis gradients of the given heat values
            at each face (3, num_pts)
    """

    u = heat_values[polys]

    # Compute outward unit normal for each face
    # 3D matrix of points in each face: n faces x 3 points per face x 3 coordinates per point
    ppts = pts[polys]

    # Compute normal vector direction
    nnfnorms = np.cross(ppts[:, 1] - ppts[:, 0],
                        ppts[:, 2] - ppts[:, 0])

    # Normalize to norm 1
    nfnorms = nnfnorms / np.sqrt(np.square(nnfnorms).sum(1))[:, np.newaxis]

    # Ensure that there are no nans
    N = np.nan_to_num(nfnorms)

    # Multiply N to the edge vector.
    fe12 = np.cross(N, ppts[:, 1] - ppts[:, 0])
    fe23 = np.cross(N, ppts[:, 2] - ppts[:, 1])
    fe31 = np.cross(N, ppts[:, 0] - ppts[:, 2])

    # Compute areas for each face
    A = face_areas(pts, polys)

    # Get transpose
    fe12T = fe12.T
    fe23T = fe23.T
    fe31T = fe31.T

    u1, u2, u3 = u.T

    # Compute gradient. Orient counter clockwise
    gradu = np.nan_to_num(ne.evaluate("(fe12T * u3 + fe23T * u1 + fe31T * u2) / (2 * A)").T)

    return gradu

def integrated_divergence(pts, polys, X):
    """
    Computes the integrated divergence of X
    :param pts: arrays of vertices
    :param polys: arrays of faces
    :param X: normalized gradient
    :return: integrated divergence at each face (num_polys, )
    """

    # 3D matrix of points in each face: n faces x 3 points per face x 3 coordinates per point
    ppts = pts[polys]

    # Compute for cotangent angles
    cots1, cots2, cots3 = cotangent_angles(pts, polys)

    # Multiply with the opposite edge
    c3 = cots3[:, np.newaxis] * (ppts[:, 1] - ppts[:, 0])
    c2 = cots2[:, np.newaxis] * (ppts[:, 0] - ppts[:, 2])
    c1 = cots1[:, np.newaxis] * (ppts[:, 2] - ppts[:, 1])

    c32 = c3 - c2
    c13 = c1 - c3
    c21 = c2 - c1

    # Compute integrated divergence
    x1 = 0.5 * (c32 * X).sum(1)
    x2 = 0.5 * (c13 * X).sum(1)
    x3 = 0.5 * (c21 * X).sum(1)

    # Create sparse matrix containing the values
    num_pt = len(pts)
    num_poly = len(polys)
    o = np.ones((num_poly,))

    s1 = sparse.coo_matrix((o, (polys[:, 0], range(num_poly))), (num_pt, num_poly)).tocsr()
    s2 = sparse.coo_matrix((o, (polys[:, 1], range(num_poly))), (num_pt, num_poly)).tocsr()
    s3 = sparse.coo_matrix((o, (polys[:, 2], range(num_poly))), (num_pt, num_poly)).tocsr()

    sm = s1.dot(x1) + s2.dot(x2) + s3.dot(x3)

    return sm


def geodesic_distance(pts, polys, verts):
    """
    Computes the geodesic distance using the heat method (Crane et al, 2012)
    :param pts: arrays of vertices
    :param polys: arrays of faces
    :param verts: array of
    :return:
    """
    num_pt = len(pts)

    C, M = laplace_operator(pts, polys)

    # Step 1: Solve the heat values u
    # time of heat evolution
    h = avg_edge_length(pts, polys)
    t = h ** 2

    # use backward Euler step
    lfac = M - t * C

    # Exclude rows with zero weight (these break the sparse LU)
    goodrows = np.nonzero(~np.array(lfac.sum(0) == 0).ravel())[0]

    # Prefactor matrices
    _rlfac_solvers = factorized(lfac[goodrows][:, goodrows])
    _nLC_solvers = factorized(C[goodrows][:, goodrows])

    # Compute u
    u0 = np.zeros((num_pt)) # initial heat values
    u0[verts] = 1.0

    goodu = _rlfac_solvers(u0[goodrows])
    u = np.zeros((num_pt,))
    u[goodrows] = goodu

    # Step 2: Compute X (normalized gradient)
    # Compute gradients at each face
    gradu = surface_gradient(pts, polys, u)

    # Compute X
    graduT = gradu.T
    gusum = ne.evaluate("sum(gradu ** 2, 1)")
    X = np.nan_to_num(ne.evaluate("-graduT / sqrt(gusum)").T)

    # Step 3: Solve the poisson equation
    # Compute integrated divergence of X at each vertex
    divx = integrated_divergence(pts, polys, X)

    # Compute phi (distance)
    goodphi = _nLC_solvers(divx[goodrows])
    phi = np.zeros((num_pt,))
    phi[goodrows] = goodphi - goodphi.min()

    # Ensure that distance is zero for selected verts
    phi[verts] = 0.0

    return phi

def color_geo(geo_dist):
    """
    Convert the distance into colors for visualization
    :param geo_dist: array of distances from the source point
    :return: array of colors corresponding to the distance
    """
    norm = mpl.colors.Normalize(vmin=0, vmax=255)
    cmap = cm.afmhot

    geo_dist2 = np.interp(geo_dist, (geo_dist.min(), geo_dist.max()), (0, 255))
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m.to_rgba(geo_dist2, bytes=True)