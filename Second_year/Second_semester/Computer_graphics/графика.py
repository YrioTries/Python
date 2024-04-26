import numpy as np
from PIL import Image, ImageOps
import math

image_matrix = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)


def dotted_line(image, x0, y0, x1, y1, count, color):
    step = 1 / count
    for t in np.arange(0, 1, step):
        x = round((1 - t) * x0 + t * x1)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color


def dotted_line_v2(image, x0, y0, x1, y1, color):
    step = 1 / math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    for t in np.arange(0, 1, step):
        x = round((1 - t) * x0 + t * x1)
        y = round((1 - t) * y0 + t * y1)
        image[y, x] = color


def x_loop_line(image, x0, y0, x1, y1, color):

    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = int(y0)
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(int(x0), int(x1)):

        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if derror > 0.5:
            derror -= 1
            y += y_update


def x_loop_line_v2_no_y_calc(image, x0, y0, x1, y1, color):

    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):

        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if derror > 0.5:
            derror -= 1
            y += y_update


def x_loop_line_v3_no_y_calc(image, x0, y0, x1, y1, color):

    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):

        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if derror > 2 * (x1 - x0) * 0.5:
            derror -= 2 * (x1 - x0) * 1
            y += y_update


def bresenham(image, x0, y0, x1, y1, color):

    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):

        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if derror > (x1 - x0):
            derror -= 2 * (x1 - x0)
            y += y_update


# for i in range(13):
# bresenham(image_matrix, 100, 100,int(100 + 95 * np.cos((2 * np.pi * i) / 13)), int(100 + 95 * np.sin((2 * np.pi * i) / 13)), 255)

# img = Image.fromarray(image_matrix, mode="RGB")

# img.save("bresenham.png")

image3d = np.zeros((2000, 2000, 3), dtype=np.uint8)
vertex_list = []
edge_list = []

f = open("model_1.obj")

tempList = []
for s in f:
    splitted = s.split()
    if splitted[0] == "v":
        vertex_list.append([float(x) for x in splitted[1:4]])
    if splitted[0] == "f":
        for i in splitted[1:4]:
            edge_list.append(int(i.split("/")[0]))

edge_list = [edge_list[i : i + 3] for i in range(0, len(edge_list), 3)]

for i in range(len(vertex_list) - 1):
    x = int(10000 * vertex_list[i][0]) + 1000
    y = int(10000 * vertex_list[i][1]) + 600
    image3d[y, x] = 255


def baricentric(x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / (
        (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    )
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / (
        (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    )
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def drawTriangle(x0, y0, x1, y1, x2, y2):
    lambda0, lambda1, lambda2 = baricentric(x0, y0, x1, y1, x2, y2)
    xmin = min(x0, x1, x2)
    xmax = max(x0, x1, x2)
    ymin = min(y0, y1, y2)
    ymax = max(y0, y1, y2)
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > 2000:
        xmax = 2000
    if ymax > 2000:
        ymax = 2000
    if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
        for i in range(len(vertex_list) - 1):
            x = int(10000 * vertex_list[i][0] + 1000)
            y = int(10000 * vertex_list[i][1] + 600)
            image3d[y, x] = 255


for j in range(len(edge_list)):
    x0 = vertex_list[edge_list[j][0] - 1][0] * 10000 + 1000
    y0 = vertex_list[edge_list[j][0] - 1][1] * 10000 + 600
    x1 = vertex_list[edge_list[j][1] - 1][0] * 10000 + 1000
    y1 = vertex_list[edge_list[j][1] - 1][1] * 10000 + 600
    x2 = vertex_list[edge_list[j][2] - 1][0] * 10000 + 1000
    y2 = vertex_list[edge_list[j][2] - 1][1] * 10000 + 600
    drawTriangle(x0, y0, x1, y1, x2, y2)
    # bresenham(image3d, x0, y0, x1, y1, 255)
    # bresenham(image3d, x1, y1, x2, y2, 255)
    # bresenham(image3d, x2, y2, x0, y0, 255)


img = Image.fromarray(image3d, mode="RGB")
img = ImageOps.flip(img)
img.show()
