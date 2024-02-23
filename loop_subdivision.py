import trimesh
from trimesh import graph, grouping
from trimesh.geometry import faces_to_edges
import numpy as np
import os

from mesh import Mesh

from itertools import zip_longest


def subdivision_loop(mesh):

    # prepare geometry for the loop subdivision
    vertices, faces = mesh.vertices, mesh.faces  # [N_vertices, 3] [N_faces, 3]
    edges, edges_face = faces_to_edges(
        faces, return_index=True)  # [N_edges, 2], [N_edges]
    edges.sort(axis=1)
    unique, inverse = grouping.unique_rows(edges)

    # split edges to interior edges and boundary edges
    edge_inter = np.sort(grouping.group_rows(edges, require_count=2), axis=1)
    edge_bound = grouping.group_rows(edges, require_count=1)

    # set also the mask for interior edges and boundary edges
    edge_bound_mask = np.zeros(len(edges), dtype=bool)
    edge_bound_mask[edge_bound] = True
    edge_bound_mask = edge_bound_mask[unique]
    edge_inter_mask = ~edge_bound_mask

    ###########
    # Step 1: #
    ###########
    # Calculate odd vertices to the middle of each edge.
    odd = vertices[edges[unique]].mean(axis=1)  # [N_oddvertices, 3]

    # connect the odd vertices with even vertices
    # however, the odd vertices need further updates over it's position
    # we therefore complete this step later afterwards.

    ###########
    # Step 2: #
    ###########
    # find v0, v1, v2, v3 and each odd vertex
    # v0 and v1 are at the end of the edge where the generated odd vertex on
    # locate the edge first
    e = edges[unique[edge_inter_mask]]
    # locate the endpoints for each edge
    e_v0 = vertices[e][:, 0]
    e_v1 = vertices[e][:, 1]

    # v2 and v3 are at the farmost position of the two triangle
    # locate the two triangle face
    edge_pair = np.zeros(len(edges)).astype(int)
    edge_pair[edge_inter[:, 0]] = edge_inter[:, 1]
    edge_pair[edge_inter[:, 1]] = edge_inter[:, 0]
    opposite_face1 = edges_face[unique]
    opposite_face2 = edges_face[edge_pair[unique]]
    # locate the corresponding edge
    e_f0 = faces[opposite_face1[edge_inter_mask]]
    e_f1 = faces[opposite_face2[edge_inter_mask]]
    # locate the vertex index and vertex location
    e_v2_idx = e_f0[~(e_f0[:, :, None] == e[:, None, :]).any(-1)]
    e_v3_idx = e_f1[~(e_f1[:, :, None] == e[:, None, :]).any(-1)]
    e_v2 = vertices[e_v2_idx]
    e_v3 = vertices[e_v3_idx]

    # update the odd vertices based the v0, v1, v2, v3, based the following:
    # 3 / 8 * (e_v0 + e_v1) + 1 / 8 * (e_v2 + e_v3)
    odd[edge_inter_mask] = 0.375 * e_v0 + \
        0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0

    ###########
    # Step 3: #
    ###########
    # find vertex neightbors for even vertices and update accordingly
    neighbors = graph.neighbors(edges=edges[unique], max_index=len(vertices))
    # convert list type of array into a fixed-shaped numpy array (set -1 to empties)
    neighbors = np.array(list(zip_longest(*neighbors, fillvalue=-1))).T
    # if the neighbor has -1 index, its point is (0, 0, 0), so that it is not included in the summation of neighbors when calculating the even
    vertices_ = np.vstack([vertices, [0.0, 0.0, 0.0]])
    # number of neighbors
    k = (neighbors + 1).astype(bool).sum(axis=1)

    # calculate even vertices for the interior case
    beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3) ** 2) / (64 * k)
    even = (
        beta[:, None] * vertices_[neighbors].sum(1)
        + (1 - k[:, None] * beta[:, None]) * vertices
    )

    ############
    # Step 1+: #
    ############
    # complete the subdivision by updating the vertex list and face list

    # the new faces with odd vertices
    odd_idx = inverse.reshape((-1, 3)) + len(vertices)
    new_faces = np.column_stack(
        [
            faces[:, 0],
            odd_idx[:, 0],
            odd_idx[:, 2],
            odd_idx[:, 0],
            faces[:, 1],
            odd_idx[:, 1],
            odd_idx[:, 2],
            odd_idx[:, 1],
            faces[:, 2],
            odd_idx[:, 0],
            odd_idx[:, 1],
            odd_idx[:, 2],
        ]
    ).reshape((-1, 3))  # [N_face*4, 3]

    odd_boundary = (vertices[edges[unique[edge_bound_mask]]].mean(axis=1))
    odd[edge_bound_mask] = odd_boundary

    i = 0  # Initialize the index for the while loop
    while i < len(vertices):
        vertex = vertices[i]
        if i in edge_bound:
            # Identify indices where current vertex is part of an edge and is a boundary
            boundary_edge_indices = [index for index, edge in enumerate(edges) if i in edge and edge_bound_mask[index]]
            
            # Collect boundary neighbors based on identified indices
            boundary_neighbors = [edge for index, edge in enumerate(edges) if index in boundary_edge_indices]
            
            if boundary_neighbors:
                # Compute the mean position of boundary neighbors
                neighbor_vertices = np.array([vertices[edge[0] if edge[1] == i else edge[1]].mean(axis=0) for edge in boundary_neighbors])
                even[i] = 0.75 * vertex + 0.125 * neighbor_vertices.sum(axis=0)

        i += 1  # Move to the next index


    # stack the new even vertices and odd vertices
    new_vertices = np.vstack((even, odd))  # [N_vertex+N_edge, 3]

    return trimesh.Trimesh(new_vertices, new_faces)


if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('data/output/cube_subdivided_4.obj')
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')
    # apply loop subdivision over the loaded mesh
    # mesh_subdivided = mesh.subdivide_loop(iterations=4)
    # print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh.export('assets/cube.obj')

    # # TODO: implement your own loop subdivision here
    iterations = 3
    mesh_subdivided = mesh
    for _ in range(iterations):
        mesh_subdivided = subdivision_loop(mesh_subdivided)
    # print the new mesh information and save the mesh
        print(f'Subdivided Mesh Info: {mesh_subdivided}')

    mesh_subdivided.show()
    mesh_subdivided.export('assets/cube_subdivided.obj')

    path = "assets/cube_subdivided.obj"
    target_v = 4
    mesh = Mesh(path)
    mesh_name = os.path.basename(path).split(".")[-2]
    simp_mesh = mesh.simplification(
        target_v=target_v)
    os.makedirs("assets/", exist_ok=True)
    simp_mesh.save(
        "assets/{}_{}.obj".format(mesh_name, simp_mesh.vs.shape[0]))
    print("[FIN] Simplification Completed!")

    mesh_decimated = trimesh.load_mesh("assets/cube_subdivided_4.obj")
    mesh_decimated.show()
