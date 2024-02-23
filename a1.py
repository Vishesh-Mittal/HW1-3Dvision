import trimesh
from loop_subdivision import subdivision_loop
import os
from mesh import Mesh
import argparse  # Use argparse for handling command line arguments

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process mesh with optional loop subdivision and simplification.")
    parser.add_argument('file_path', nargs='?', default='cube', help='Path to the mesh file. Defaults to generating a cube.')
    parser.add_argument('--iterations', type=int, default=3, help='Number of loop subdivision iterations. Default is 3.')
    parser.add_argument('--vertices', type=int, default=4, help='Desired number of vertices after simplification. Default is 4.')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Ensure the assets directory exists
    os.makedirs("assets/", exist_ok=True)
    
    # Load the mesh from file or generate a cube if no file is specified
    if args.file_path != 'cube':
        try:
            initial_mesh = trimesh.load_mesh(args.file_path)
        except Exception as e:
            print(f"Failed to load mesh from {args.file_path}. Error: {e}")
            return
    else:
        initial_mesh = trimesh.creation.box(extents=[1, 1, 1])
        initial_mesh.export('assets/original_cube.obj')

    print(f'Initial Mesh Details: {initial_mesh}')
    
    # Apply loop subdivision
    refined_mesh = initial_mesh
    for iteration in range(args.iterations):
        refined_mesh = subdivision_loop(refined_mesh)
        print(f'Refined Mesh Details after iteration {iteration + 1}: {refined_mesh}')

    # Export the refined mesh
    mesh_label = 'subdivision_loop'
    if args.file_path != 'cube':
        mesh_label += '_' + os.path.splitext(os.path.basename(args.file_path))[0]
    else:
        mesh_label += '_cube'
    refined_mesh.export(f'assets/{mesh_label}.obj')

    refined_mesh_path = f"assets/{mesh_label}.obj"
    refined_mesh = Mesh(refined_mesh_path)
    
    # Simplify the mesh
    simplified_mesh = refined_mesh.simplification(target_v=args.vertices)
    
    simplified_mesh_path = f"assets/{mesh_label}_{len(simplified_mesh.vs)}.obj"
    simplified_mesh.save(simplified_mesh_path)
    print("[COMPLETE] Mesh simplification process finished.")

    # Load the simplified mesh for further operations or visualization
    final_mesh = trimesh.load_mesh(simplified_mesh_path)

if __name__ == '__main__':
    main()
