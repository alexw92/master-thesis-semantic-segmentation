from matplotlib import pyplot
from shapely.geometry import Point, Polygon
import numpy as np
#
# fig = pyplot.figure()
# ax = fig.add_subplot(121)


def color_polygon(polygon, size=5, poly_class=1, im_mask=None,):
    """
    Fills the polygon `polygon` with values `poly_class` leaving holes uncolored
    Returns the image with the colored polygones
    :param polygon: the polygon to be colored
    :param poly_class: the class/color used for coloring
    :param im_mask:
    :param size:
    :return:
    """
    if im_mask is None:
        im_mask = np.zeros(shape=(size, size), dtype='int8')
    print(size)
    left, right, up, down = size, 0, size, 0
    for p in polygon.exterior.coords:
        if p[0]<left:
            left = round(p[0])
        if p[0]>right:
            right = round(p[0])
        if p[1]<down:
            down = round(p[1])
        if p[1]>up:
            up = round(p[1])
    print(str(left)+' '+str(right)+' '+str(down)+' '+str(up))
    # get Xs and Ys values from points being part of the polygone
    XYs = [(x, y) for x in range(left, right) for y in range(down, up) if Point(x, y).intersects(polygon)]
    Xs = [round(xy[0]) for xy in XYs]
    Ys = [round(xy[1]) for xy in XYs]
    # assignment to multiple points
    im_mask[Xs, Ys] = poly_class
    pyplot.imshow(im_mask, origin='lower')
    print('poly processed')
    return im_mask


# ext = [(0, 0), (0, 4), (4, 4), (4, 0), (0, 0)]
# int = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)][::-1]
# polygon = Polygon(ext, [int])
#
#
# ar = color_polygon(polygon, poly_class=1, im_mask=None, size=10)
# print((2, 2) in ar)  # should be false as (2, 2) is part of the whole
# pyplot.show()
