# 3D Mesh Processing Toolkit

This toolkit provides an implementation of two advanced 3D mesh processing techniques: Loop Subdivision and Quadric Error Metrics (QEM) Simplification. It is designed to help with the refinement and simplification of 3D mesh models, making it an essential tool for graphics programmers, modelers, and enthusiasts interested in 3D modeling.

## Features

- **Loop Subdivision**: Refines meshes by subdividing triangles into smaller triangles, increasing the mesh's detail and smoothness.
- **QEM Simplification**: Reduces the complexity of a mesh by decreasing the number of vertices, while preserving the model's overall shape and features.

## Project Structure

- `a1.py`: The main script that processes a mesh with options for loop subdivision and simplification.
- `loop_subdivision.py`: Contains the implementation of the Loop Subdivision algorithm.
- `mesh.py`: Defines the mesh structure used for QEM Simplification.
- `assets/`: Contains generated 3D objects and sample meshes for processing.

## Installation

To use this toolkit, you need Python installed on your machine. Clone this repository to your local machine, and you're ready to process meshes.

```bash
git clone https://your-repository-url.git
cd your-project-directory
```
### Usage

The toolkit is executed through the command line with various options for processing the meshes. Below is the basic syntax:

```bash
python a1.py [file_path] [options]
```

## Command line arguments

- file_path: Path to the mesh file you wish to process. If not provided, defaults to generating a cube.
- iterations: Number of loop subdivision iterations to apply. Default is 3.
- vertices: Desired number of vertices after simplification. Default is 4.

## Examples

To process a mesh with default settings (generate a cube and apply 3 iterations of loop subdivision):

```bash
python a1.py
```

To process a mesh file with 5 iterations of loop subdivision:

```bash
python a1.py path/to/your/mesh.obj --iterations 5
```

To simplify a mesh to a desired number of vertices:

```bash
python a1.py path/to/your/mesh.obj --vertices 100
```
