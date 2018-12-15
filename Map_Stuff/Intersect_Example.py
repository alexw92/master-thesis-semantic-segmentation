from matplotlib import pyplot
from shapely.geometry import Point, LineString
from descartes import PolygonPatch
from shapely.ops import cascaded_union

#from figures import SIZE, BLUE, GRAY

fig = pyplot.figure(1,figsize=(10,10), dpi=90)
polygons = []
a = Point(1, 1).buffer(1.5)
b = Point(2, 1).buffer(1.5)
# line to polygon!
line = LineString([(0,0), (0,1), (0,2), (1,2),(3,3)])
polygons.append(line.buffer(0.1))
ps = cascaded_union(polygons)
print(ps)
# 1
ax = fig.add_subplot(121)

patch1 = PolygonPatch(ps, fc=(0.10,0.10,0.10), ec=(0.10,0.10,0.10), alpha=0.2, zorder=1)
ax.add_patch(patch1)
patch2 = PolygonPatch(b, fc=(0.10,0.10,0.10), ec=(0.10,0.10,0.10), alpha=0.2, zorder=1)
ax.add_patch(patch2)
c = a.intersection(b)
patchc = PolygonPatch(c, fc=(0,0,0.5), ec=(0,0,0.5), alpha=0.5, zorder=2)
ax.add_patch(patchc)

ax.set_title('a.intersection(b)')

xrange = [-1, 4]
yrange = [-1, 3]
ax.set_xlim(*xrange)
# thx to https://stackoverflow.com/a/13318111/8862202
ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
ax.set_aspect(1)

#2
ax = fig.add_subplot(122)

patch1 = PolygonPatch(a, fc=(0.10,0.10,0.10), ec=(0.10,0.10,0.10), alpha=0.2, zorder=1)
ax.add_patch(patch1)
patch2 = PolygonPatch(b, fc=(0.10,0.10,0.10), ec=(0.10,0.10,0.10), alpha=0.2, zorder=1)
ax.add_patch(patch2)
c = a.symmetric_difference(b)

if c.geom_type == 'Polygon':
    patchc = PolygonPatch(c, fc=(0,0,0.5), ec=(0,0,0.5), alpha=0.5, zorder=2)
    ax.add_patch(patchc)
elif c.geom_type == 'MultiPolygon':
    for p in c:
        patchp = PolygonPatch(p, fc=(0,0,0.5), ec=(0,0,0.5), alpha=0.5, zorder=2)
        ax.add_patch(patchp)

ax.set_title('a.symmetric_difference(b)')

xrange = [-1, 4]
yrange = [-1, 3]
ax.set_xlim(*xrange)
# thx to https://stackoverflow.com/a/13318111/8862202
ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
ax.set_aspect(1)

pyplot.show()

