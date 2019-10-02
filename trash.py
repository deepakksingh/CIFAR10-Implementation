from collections import namedtuple

Point = namedtuple("Point",['x', 'y'])
p = Point(x=10, y=1)
print(p)
print(p.x)
x, y = p
print(x, y)