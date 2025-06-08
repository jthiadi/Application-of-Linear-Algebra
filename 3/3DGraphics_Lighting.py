import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import os

# [TODO]
def compute_lookat(azim: float, elev: float):
    """
        Compute the look at vector. (Definition in Figure 3)
        azim: float, degree in [-180, 180],
        elev: float, degree in [-180, 180]
    """

    # For the definition of azim and elev, check
    # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html

    # convert degrees to radians
    azim_rad = np.radians(azim)
    elev_rad = np.radians(elev)

    a = np.cos(elev_rad)  # projection of the vector onto the XY-plane
    b = np.sin(elev_rad)  # vertical component

    # compute the look-at vector
    lookat = np.array([a * np.cos(azim_rad),  # X-component
                       a * np.sin(azim_rad),  # Y-component
                       b])                    # Z-component

    # normalize the look-at vector
    lookat = lookat / np.linalg.norm(lookat)

    return lookat

    # [TODO]
def compute_normal(P1: tuple, P2: tuple, P3: tuple):
    """
        Compute the normal vector, given P1, P2, P3 in counter-clockwise order.
    """

    # compute vectors representing edges of the triangle
    v1 = np.array(P2) - np.array(P1)  # vector from P1 to P2
    v2 = np.array(P3) - np.array(P1)  # vector from P1 to P3

    # compute the cross product of the edges to get the normal vector
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    return normal

    # [TODO]
def visible(face_normal: np.ndarray, lookat: np.ndarray):
    """
        Given a normal vector of a face, determine if the face (outward-facing side) is visible
    """

    # use the normal vector of the triangle and the lookat direction
    dot_product = np.dot(face_normal, lookat)

    # the face is visible if the dot product is positive
    return dot_product > 0

    def compute_intensity(face_normal: np.ndarray, lookat: np.ndarray, lightsource: np.ndarray):
    """
    Given normal vector of a face (face_normal), viewing vector (lookat) and lightsource (lightsource),
    compute the specular intensity.
    """
    # Compute the reflection vector R
    reflect = (2 * np.dot(face_normal, lightsource) * face_normal) - lightsource

    # constants, adjust accordingly
    n = 0.3                 # shininess
    min_intensity = 0.1     # minimal intensity so it won't be black

    # Compute the dot product of R and V
    cos_beta = np.dot(reflect, lookat)

    normalize_range = (cos_beta + 1) / 2 # range from 0 to 1 so no black

    intensity = max(min_intensity, normalize_range) ** n
    # Compute the intensity using the Phong reflection model
    #if cos_beta > 0:
    #    intensity = cos_beta  ** n
    #else:
    #    intensity = 0.0  # no specular reflection

    return intensity

def read_obj_file(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(index.split('/')[0]) for index in line.strip().split()[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)    

# Read .obj file
obj_file_path = 'dodecahedron.obj'
#obj_file_path = 'icosahedron_modified.obj'
vertices, faces = read_obj_file(obj_file_path)

# Center the object
center = np.mean(vertices, axis=0)
vertices = vertices - center

# Get object dimensions for plotting
max_x = np.max(vertices[:, 0])
min_x = np.min(vertices[:, 0])
max_y = np.max(vertices[:, 1])
min_y = np.min(vertices[:, 1])
max_z = np.max(vertices[:, 2])
min_z = np.min(vertices[:, 2])

#basecolor = np.array([0, 1, 0.7])
basecolor = np.array([0.8, 0.5, 1.0]) # changed the model to light purple
#basecolor = np.random.rand(len(faces), 3)  # random RGB values for each face

# list to store frames for the GIF
frames = []

# number of frames for the animation
num_frames = 100

# loop to create frames with varying light source positions
for i in range(num_frames):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_axis_off()
    ax.set_xlim(min_x, max_x), ax.set_ylim(min_y, max_y), ax.set_zlim(min_z, max_z)
    ax.view_init(azim=60, elev=30)

    lookat = compute_lookat(ax.azim, ax.elev)

    phi = 2 * np.pi * i / num_frames  # angle for light source movement
    lightsource = np.array([np.cos(phi), np.sin(phi), np.sin(phi)])
    lightsource = lightsource / np.linalg.norm(lightsource)

    # Plot faces
    for face in faces:
        P1, P2, P3 = vertices[face[0] - 1], vertices[face[1] - 1], vertices[face[2] - 1]
        face_normal = compute_normal(P1, P2, P3)

        if visible(face_normal, lookat):
            specular_intensity = compute_intensity(face_normal, lookat, lightsource)
            color = basecolor * specular_intensity
            color = np.clip(color, 0.0, 1.0)

            ax.add_collection3d(Poly3DCollection(
                [np.array([P1, P2, P3])],
                color=color, edgecolor='white', linewidth=1
            ))

    # save the current frame to a temporary file
    fname = f'temp_frame_{i}.png'
    plt.savefig(fname)
    plt.close(fig)  # close the figure to release resources

    # read the saved frame and append it to the frames list
    frames.append(imageio.imread(fname))

    # remove the temporary frame file
    os.remove(fname)

# create GIF from the collected frames
studentID = '112006234'  # replace with your student ID
gif_fname = f'HW3_{studentID}.gif'
imageio.mimsave(gif_fname, frames, fps=10)  # adjust fps as necessary

first_frame_fname = f'HW3_{studentID}.png'
imageio.imwrite(first_frame_fname, frames[0]) # save first frame

from collections import defaultdict

def load_obj(file_path):
    vertices = []
    faces = []
    vertex_normals = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            values = line.split()
            if not values:
                continue

            if values[0] == 'v':
                vertices.append([float(x) for x in values[1:4]])
            elif values[0] == 'vn':
                vertex_normals.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                vertex_indices = []
                normal_indices = []

                for vertex in values[1:]:
                    # deal with v//vn
                    if '//' in vertex:
                        v_idx, _, n_idx = vertex.split('/')
                        vertex_indices.append(int(v_idx) - 1)
                        normal_indices.append(int(n_idx) - 1)
                    # deal with v
                    else:
                        vertex_indices.append(int(vertex) - 1)

                faces.append(vertex_indices)

    return np.array(vertices), np.array(faces), np.array(vertex_normals)

    def compute_vertex_normals(vertices, faces):
    # Initialize an array to store normals for each vertex
    vertex_normals = np.zeros_like(vertices)

    for face in faces:
        # Extract the vertices of the face
        P1, P2, P3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

        # Compute the face normal
        face_normal = np.cross(P2 - P1, P3 - P1)
        face_normal /= np.linalg.norm(face_normal)

        # Accumulate the face normal to each vertex in the face
        vertex_normals[face[0]] += face_normal
        vertex_normals[face[1]] += face_normal
        vertex_normals[face[2]] += face_normal

    # Normalize the accumulated vertex normals
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / norms  # Avoid division by zero

    return vertex_normals

obj_path = "icosahedron_modified.obj"
vertices, faces, vn = load_obj(obj_path)
vertex_normals = compute_vertex_normals(vertices, faces)

print("vertex normal：")
for i, normal in enumerate(vertex_normals):
     print(f"v{i+1:02d} normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")

if (np.allclose(vertex_normals, vn)):
    print("PASS")
else :
    print("ERROR")