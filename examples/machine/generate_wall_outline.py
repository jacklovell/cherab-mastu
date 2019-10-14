"""
This script generates an optical outline for the MAST-U vessel.

To do this, load in the full 3D CAD mesh of MAST-U. Pick a particular
toroidal angle. Then fire rays from the magnetic axis in the poloidal
plane at a range of poloidal angles and record where they hit the
walls. Repeat the process for rays fired from the centre of the upper
and lower divertor chambers, the upper and lower throats, upper and
lower cryo area behind T5 and between T5 and the baffle to ensure there
are no shadowed parts of the vessel. Finally, clean up the resulting
points to form a series of simplified paths giving the outline of the
vessel and coil cans.
"""

from concurrent.futures import ProcessPoolExecutor
import datetime
from itertools import chain
import math
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from raysect.core import World, Ray, Point2D, Point3D, Vector3D
from raysect.core.math import rotate_y, rotate_z, translate
from raysect.optical import AbsorbingSurface, NullMaterial
from raysect.primitive import Cone, Cylinder, Intersect, Subtract
from cherab.mastu.machine import import_mastu_mesh


class IgnoreSurface(AbsorbingSurface):
    """A material to indicate an intersection point which should be ignored."""


def calculate_hit_points(world, origin_point, poloidal_angles, toroidal_angle):
    """Calculate the hit points on the vessel

    :param World world: the world containing the vessel
    :param Point2D origin_point: the position in the R-Z plane the rays from
    :param poloidal_angles: an iterable of angles to fire the rays
    :param toroidal_angle: the toroidal angle (in degrees) of the poloidal slice

    :return list[Point2D]: a list of the hit points on the vessel in the R-Z plane
    """
    # Convert to 3D origin point
    toroidal_transform = rotate_z(toroidal_angle)
    origin_point = Point3D(origin_point.x, 0, origin_point.y).transform(toroidal_transform)
    intersection_points = []
    for angle in poloidal_angles:
        direction = Vector3D(0, 0, 1).transform(
            toroidal_transform * translate(*origin_point) * rotate_y(angle)
        )
        origin = origin_point
        while True:
            intersection = world.hit(Ray(origin, direction))
            if intersection is None:
                break

            if isinstance(intersection.primitive.material, IgnoreSurface):
                intersection = None
                break

            if isinstance(intersection.primitive.material, NullMaterial):
                hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
                # apply a small displacement to avoid infinite self collisions due to numerics
                ray_displacement = 0.001
                origin = hit_point + direction * ray_displacement
                continue
            else:
                break
        if intersection is not None:
            # Convert to R-Z plane
            hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
            point_r = math.hypot(hit_point.x, hit_point.y)
            point_z = hit_point.z
            intersection_points.append(Point2D(point_r, point_z))
    return intersection_points


def point2d_to_2darray(list_of_points):
    """Convert a list of Point2D to an Nx2 numpy array

    :param list[Point2D] list_of_points: the list of points to convert
    :return ndarray points_array: the list of points as an Nx2 array
    """
    return np.fromiter(chain.from_iterable(list_of_points), float).reshape((-1, 2))


def remove_unwanted_points(hit_points):
    """
    Remove hit points in regions which definitely shouldn't form the limiter

    This prevents the KD tree going down the wrong branch and following
    hit points we're really not interested in.

    :param list[Point2D] hit_points: a list of hit points
    :return ndarray: hit points with unwanted regions removed

    The regions where hit points are discarded are as follows:
    - The outer cylinder between the upper and lower HE* ports
    - P4 cases
    - The end plates at radii smaller than th edge of T5
    - P5 mounting brackets
    - The insides of P5 if there is any P5 armour
    - The insides of the lower ELM coil
    """
    hit_points = point2d_to_2darray(hit_points)
    hit_r = hit_points[:, 0]
    hit_z = hit_points[:, 1]
    end_plate_region = ((hit_r < 1.41) & (abs(hit_z) > 2.15))
    outer_cylinder_region = ((hit_r > 1.99) & (abs(hit_z) < 1.0))
    p4_vertical_region = (abs(hit_r - 1.29) < 0.01) & (abs(hit_z) > 1.05) & (abs(hit_z) < 1.30)
    p4_horizontal_region = (hit_r > 1.40) & (hit_r < 1.70) & (abs(abs(hit_z) - 1.00) < 0.01)
    p5_mounting_bracket = (hit_r > 1.77) & (abs(hit_z) < 0.8)
    inside_elm_coil_76 = (hit_r > 1.415) & (hit_z < -0.705) & (hit_z > -0.725)  # Also 346
    inside_elm_coil_166 = (hit_r > 1.423) & (hit_z < -0.692) & (hit_z > -0.705)  # Also 256
    inside_elm_coil = inside_elm_coil_76 | inside_elm_coil_166
    # If there is P5 armour, remove inner face of P5
    if np.any((abs(hit_r - 1.455) < 0.001) & (hit_z > 0.20) & (hit_z < 0.45)):
        p5u_inner = (abs(hit_r - 1.554) < 0.001) & (hit_z > 0.25) & (hit_z < 0.45)
    else:
        p5u_inner = False
    if np.any((abs(hit_r - 1.455) < 0.001) & (hit_z < -0.20) & (hit_z > -0.45)):
        p5l_inner = (abs(hit_r - 1.554) < 0.001) & (hit_z < -0.25) & (hit_z > -0.45)
    else:
        p5l_inner = False

    accept = ~(end_plate_region | outer_cylinder_region | p4_vertical_region | inside_elm_coil
               | p4_horizontal_region | p5_mounting_bracket | p5u_inner | p5l_inner)
    return hit_points[accept]


def reorder_hit_points(hit_points, starting_point, max_distance=0.01):
    """
    Arrange the list of hit points so that nearest neighbours are adjacent.

    This function takes an array of hit points and reorders them to
    place nearest neighbours next to each other. This then enables the
    hit points to be used to describe a simple polygon.

    :param list[Point2D] hit_points: a list of hit points
    :param Point2D starting_point: where to start ordering the points
    :return ndarray ordered_points: the array of hit points, reordered
    """
    hit_points = point2d_to_2darray(hit_points)
    hit_points = remove_unwanted_points(hit_points)
    n_points, n_dim = hit_points.shape
    if n_dim != 2:
        raise ValueError("hit_points should be an Nx2 array")
    tree = cKDTree(hit_points)
    current_point = tree.data[tree.query(list(starting_point))[1]]
    ordered_points = [tuple(current_point)]
    ordered_points_set = set(ordered_points)  # For faster `point in points` test
    for _ in range(1, n_points):
        next_distance, next_int = tree.query(current_point, k=1500, distance_upper_bound=max_distance)
        for distance, index in zip(next_distance, next_int):
            if index == n_points:
                break
            if distance > max_distance:
                break
            next_point = tuple(tree.data[index])
            # Don't include points already in the set
            if next_point in ordered_points_set:
                continue
            current_point = next_point
            ordered_points.append(current_point)
            ordered_points_set.add(current_point)
            break
    return np.asarray(ordered_points)


def clean_and_simplify(hit_points):
    """
    Clean up any remaining self intersections and simplify the path

    :param ndarray hit_points: an Nx2 array of hit points
    :return ndarray cleaned_points: an Nx2 array of a clean polygon
    """
    poly = Polygon(hit_points)
    # Clean the polygon using Polygon.buffer(0). If multiple polygons are
    # returned, pick the one with the most points as this is likely to be
    # the main polygon of interest.
    try:
        poly_valid = max(poly.buffer(0), key=lambda p: len(p.exterior.xy[0]))
    except TypeError:
        poly_valid = poly.buffer(0)
    poly_simplified = poly_valid.simplify(1e-4)
    return np.array(poly_simplified.exterior.xy).T


def reorder_all_hit_points(hit_points):
    """
    Arrange the list of hit points so that nearest neighbours are adjacent.

    This function takes an array of hit points and reorders them to
    place nearest neighbours next to each other. This then enables the
    hit points to be used to describe a simple polygon.

    This function calls reorder_hit_points with two starting points,
    above and below the mid plane, to ensure the KD Tree steps are
    symmetric.

    :param list[Point2D] hit_points: a list of hit points
    :return ndarray ordered_points: the array of hit points, reordered
    """
    hit_points_array = point2d_to_2darray(hit_points)
    hit_points_abovemid = hit_points_array[hit_points_array[:, 1] > 0]
    hit_points_belowmid = hit_points_array[hit_points_array[:, 1] < 0]
    ordered_points_lower = reorder_hit_points(hit_points_belowmid,
                                              Point2D(1.75, -0.25), 0.5)
    ordered_points_upper = reorder_hit_points(hit_points_abovemid,
                                              Point2D(1.75, 0.25), 0.5)
    ordered_points = np.concatenate((ordered_points_lower, ordered_points_upper[::-1]))
    return ordered_points


def plot_hit_points(hit_points, ax=None, **plot_kwargs):
    """Plot the hit points

    :param list[Point2D] hit_points: the intersections of rays with the vessel
    :param plot_kwargs: keyword arguments to pass to Matplotlib's plot()
    """
    if ax is None:
        _, ax = plt.subplots()
    hitr, hitz = point2d_to_2darray(hit_points).T
    ax.plot(hitr, hitz, **plot_kwargs)
    ax.axis('equal')
    return ax


def calculate_all_hit_points(world, toroidal_angle, rays_per_origin=5000):
    """
    Calculate hit points by firing rays from multiple origins

    :param toroidal_angle: the toroidal angle of the RZ plane to image
    :param rays_per_origin: number of rays to fire from each starting point

    :return hit_points: all points where rays hit a surface, unordered
    """
    origin_points = dict(
        magnetic_axis=Point2D(1, 0),
        upper_divertor_centre=Point2D(1.2, 1.8),
        lower_divertor_centre=Point2D(1.2, -1.8),
        upper_throat=Point2D(0.75, 1.6),
        lower_throat=Point2D(0.75, -1.6),
        upper_cryo=Point2D(1.8, 2.1),
        lower_cryo=Point2D(1.8, -2.1),
        below_t5u=Point2D(1.62, 1.64),
        above_t5l=Point2D(1.62, -1.64),
    )
    poloidal_angles = np.linspace(0, 360, rays_per_origin, endpoint=False)
    hit_points = np.concatenate(
        [calculate_hit_points(world, point, poloidal_angles, toroidal_angle)
         for point in origin_points.values()]
    )
    return hit_points


def save_outlines(outlines):
    """
    Save the outlines for all 12 sectors to netCDF

    :param outlines: a dictionary of (toroidal_angle=outline) pairs
    """
    now = datetime.datetime.now().replace(microsecond=0)
    with nc.Dataset("optical_limiter.nc", "w") as root_group:
        root_group.creationDate = str(now.date())
        root_group.coordinateSystem = "cylindrical"
        root_group.device = "MAST-U"
        root_group.shotRangeStart = 50000
        root_group.shotRangeEnd = 9999999
        root_group.createdBy = "jlovell"
        root_group.system = "limiter"
        root_group.class_ = "passive structure"
        root_group.units = "SI, degrees"
        root_group.version = 0
        root_group.revision = 1
        root_group.conventions = "MAST-U MetaData 0.1"
        root_group.status = "development"
        root_group.releaseDate = str(now.date())
        root_group.releaseTime = str(now.time())
        root_group.creatorCode = "python cherab_mastu/examples/machine/generate_wall_outline.py"
        limiter_group = root_group.createGroup("/limiter")
        optical_group = limiter_group.createGroup("/optical")
        optical_group.createDimension("singleDim", 1)
        for angle, outline in outlines.items():
            # The machine description follows the convention that phi=0 is the x
            # axis (due East on MAST-U, between sectors 3 and 4)
            sector = ((360 - angle) // 30 + 3) % 12 + 1
            r, z = outline.T
            limiter_polygon_dtype = np.dtype([
                ('name', '<S50'),
                ('refFrame', '<S50'),
                ('version', '<S10'),
                ('phi_cut', np.float32),
                ('R', np.float32, (len(r),)),
                ('Z', np.float32, (len(z),)),
            ])
            nc_dtype = optical_group.createCompoundType(
                limiter_polygon_dtype, "S{:02d}_LIMITER".format(sector)
            )
            limiter_polygon = np.empty(1, dtype=limiter_polygon_dtype)
            limiter_polygon[0] = ("Optical Limiter", 'Machine', '0.1', angle, r, z)
            optical_limiter = optical_group.createVariable(
                "S{:02d}".format(sector), nc_dtype, "singleDim"
            )
            optical_limiter[:] = limiter_polygon


def make_world():
    """
    Produce the world, containing the vessel and bounding primitives.
    """
    world = World()
    import_mastu_mesh(world, override_material=AbsorbingSurface())
    # Add a cylinder to cover up ports on the UEP and top and bottom
    # of the main vessel
    Cylinder(radius=2.0, height=4.4, transform=translate(0, 0, -2.2),
             name="Enclosing cylinder", parent=world, material=AbsorbingSurface())
    # Block off the main-chamber-facing surfaces of the baffles
    inclusion_cylinder = Cylinder(radius=2.1, height=1.5, transform=translate(0, 0, 1e-6))
    outer_baffle_block = Cone(radius=2.1, height=3)
    inner_baffle_block = Cone(radius=2.1 * 0.99, height=3 * 0.99)
    baffle_block = Subtract(outer_baffle_block, inner_baffle_block)
    Intersect(baffle_block, inclusion_cylinder, name="Upper baffle block",
              material=IgnoreSurface(), parent=world)
    Intersect(baffle_block, inclusion_cylinder, name="Lower baffle block",
              material=IgnoreSurface(), parent=world, transform=rotate_y(180))
    return world


def main():
    """Entry point of the program"""
    world = make_world()
    # The machine description follows the convention that phi=0 is the x axis
    # (due East on MAST-U, between sectors 3 and 4)
    toroidal_angles = range(16, 360, 30)
    hit_points = [calculate_all_hit_points(world, angle) for angle in toroidal_angles]
    ordered_hit_points = []
    with ProcessPoolExecutor(max_workers=len(toroidal_angles)) as ex:
        for ordered in ex.map(reorder_all_hit_points, hit_points):
            ordered_hit_points.append(ordered)
    outlines = {}
    for angle, hit, ordered in zip(toroidal_angles, hit_points, ordered_hit_points):
        cleaned = clean_and_simplify(ordered)
        _, ax = plt.subplots()
        plot_hit_points(hit, ax=ax, marker='.', markersize=2, linestyle='none')
        plot_hit_points(cleaned, ax=ax, marker='x', linestyle='-', alpha=0.7)
        ax.set_title("Toroidal angle {}".format(angle))
        outlines[angle] = cleaned
    save_outlines(outlines)
    input("Press enter to plot...")
    plt.show()


if __name__ == "__main__":
    main()
