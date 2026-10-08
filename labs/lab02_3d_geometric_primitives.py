import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon


num_points = 5
inner_radius = 0.4
outer_radius = 1.0
thickness = 0.3
top_z = thickness / 2
bottom_z = -thickness / 2


top_vertices = []
bottom_vertices = []

for i in range(num_points * 2):
    angle_rad = i * np.pi / num_points
    r = outer_radius if i % 2 == 0 else inner_radius
    x = r * np.cos(angle_rad)
    y = r * np.sin(angle_rad)
    top_vertices.append([x, y, top_z])
    bottom_vertices.append([x, y, bottom_z])


faces = []
for i in range(len(top_vertices)):
    next_i = (i + 1) % len(top_vertices)
    v1 = top_vertices[i]
    v2 = top_vertices[next_i]
    v3 = bottom_vertices[next_i]
    v4 = bottom_vertices[i]
    faces.append([v1, v2, v3, v4])


for i in range(len(top_vertices) - 1):
    faces.append([[0, 0, top_z], top_vertices[i], top_vertices[i + 1]])
faces.append([[0, 0, top_z], top_vertices[-1], top_vertices[0]])


for i in range(len(bottom_vertices) - 1):
    faces.append([[0, 0, bottom_z], bottom_vertices[i + 1], bottom_vertices[i]])
faces.append([[0, 0, bottom_z], bottom_vertices[0], bottom_vertices[-1]])


light_dir = np.array([5.0, 5.0, 5.0])
light_dir = light_dir / np.linalg.norm(light_dir)


face_colors = []
for face in faces:
    v1, v2, v3 = face[0], face[1], face[2]


    u = np.subtract(v2, v1)
    v = np.subtract(v3, v1)
    normal = np.cross(u, v)
    norm = np.linalg.norm(normal)
    if norm != 0:
        normal = normal / norm


    brightness = max(np.dot(normal, light_dir), 0)


    specular = np.power(brightness, 20)


    ambient = 0.2
    diffuse = 0.6 * brightness
    highlight = 0.8 * specular
    intensity = min(ambient + diffuse + highlight, 1.0)


    color = (intensity, intensity * 0.84, 0)
    face_colors.append(color)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d', facecolor='black')
ax.axis('off')

poly = Poly3DCollection(faces, facecolors=face_colors, edgecolor='black', linewidths=0.3)
ax.add_collection3d(poly)

ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-0.5, 0.5])

plt.tight_layout()
plt.show()
