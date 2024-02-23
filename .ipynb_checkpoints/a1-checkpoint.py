import trimesh
from loop_subdivision import subdivision_loop
import os
from mesh import Mesh
import numpy as np

if __name__ == '__main__':

    os.makedirs("assets/", exist_ok=True)
    
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    print(f'Mesh Info: {mesh}')
    
    #simp_mesh.save(save_path)
    mesh.export('assets/cube.obj')

    # # TODO: implement your own loop subdivision here
    iterations = 3
    mesh_subdivided = mesh
    for _ in range(iterations):
        mesh_subdivided = subdivision_loop(mesh_subdivided)
    # print the new mesh information and save the mesh
        print(f'Subdivided Mesh Info: {mesh_subdivided}')

    #mesh_subdivided.show()
    mesh_subdivided.export('assets/cube_subdivided.obj')

    path = "assets/cube_subdivided.obj"
    target_v = 4
    mesh = Mesh(path)
    mesh_name = os.path.basename(path).split(".")[-2]
    simp_mesh = mesh.simplification(
        target_v=target_v)
    
    simp_mesh.save("assets/{}_{}.obj".format(mesh_name, simp_mesh.vs.shape[0]))
    print("[FIN] Simplification Completed!")

    mesh_decimated = trimesh.load_mesh("assets/cube_subdivided_4.obj")
    #mesh_decimated.show()
