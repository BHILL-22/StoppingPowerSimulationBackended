"""Options relating to determining the stopping power over a path"""
from stopping_power_ml.rc import *
from scipy.integrate import quad
import numpy as np
from itertools import product

def calc_angle(a, b):
    return np.arccos(np.clip(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))/np.pi*180

def compute_min_perodic_vector(lattice, vdir, max_search = 10):

    if np.allclose(vdir, np.round(vdir), atol = 1e-6):
        # Use gcd if the vector has all integer elements
        vdir = np.round(vdir).astype(int)

        # Determine the shortest-possible lattice vector
        g = np.gcd.reduce(vdir)
        min_vec = vdir // g

        logging.info(f"Minimum periodic primivitive cell lattice vector is {min_vec}")

        # Compute the path length
        return min_vec
    
    # if the vector has irrational elements, approximately determine the shortest possible lattice vector
    else:
        angle_tol = 0.6 # degree
        vdir_unit = vdir/np.linalg.norm(vdir)
        min_vec = None
        min_proj_len = np.inf
        min_angle = np.inf
        min_R = None

        for coeffs in product(range(-max_search, max_search), repeat = 3):
            if coeffs == (0, 0, 0):
                continue

            R = np.array(coeffs) 

            proj_len = np.dot(R, vdir_unit)
            angle = calc_angle(R, vdir_unit)

            if (proj_len > 0) and (angle < angle_tol):
                if (proj_len < min_proj_len):
                    min_proj_len = proj_len
                    min_R = R
                    min_vec = vdir_unit*np.linalg.norm(R)
                    min_angle = angle

        if (min_vec is not None):
            logging.info(f"The angle between the velocity direction, {vdir}, and the selected primitive cell vector for minimum periodic distance, {min_R}, is {min_angle} degree.")
            return min_vec
        else:
            if (max_search < 60):
                return compute_min_perodic_vector(lattice, vdir, max_search + 10)
            else:
                raise ValueError("No valid lattice vector found to be close to the velocity direction within {angle_tol} degree")

def compute_trajectory(cell, vdir):
    """Given a contravariant vector, compute the minimum path length needed to determine stopping power.

    The contravariant vector should be for the conventional cell of the structure. The path will be determined using
    the primitive cell.

    :param cell: [Cell], Cell object defined in Cell.py
    :param vdir: [int], velocity direction
    :return: ndarray, 3x1 array defining the shortest path the covers an entire path"""

    # Map the contravariant vector of the conventional cell to that of the primitive cell
    prim_vector = np.dot(cell.conv_to_prim, np.array(vdir))

    logging.info(f"Contravariant velocity vector in primitive cell: {prim_vector}")

    min_vec = compute_min_perodic_vector(cell.prim_strc.lattice.matrix, prim_vector)

    angle = calc_angle(prim_vector, min_vec)

    traj = np.dot(cell.prim_strc.lattice.matrix, min_vec)

    logging.info(f"traj direction in Cartesian coordinate {traj} and length {np.linalg.norm(traj)} bohr")

    return traj

def _find_near_hits(cell, start_pos, vector, threshold, estimate_extrema=False):
    """Determine the positions of near-hits along a trajectory.

    These positions are locations where there is a 'spike' in the force acting on the projectile, which cause
    issues with the integration scheme. It there are two peaks for each near-hit: one as it approaches and
    one as it departs (at the point of closest path the force from the largest contribution to force, ion-ion
    repulsion, is zero).

    If you set `estimate_estimate_extrema` to True, this code will return the estimated positions of the two
    peaks. If False, this function returns the position of closest path.

    The positions are returned in fractional displacement along a trajectory, where 0 is the starting point
    and 1 is the first point at which the trajectory starts to repeat (due to symmetry).

    :param cell: [Cell], Cell object defined in Cell.py
    :param start_pos: [float], starting point in conventional cell fractional coordinates
    :param traj: [float]*3, traj of the minimum periodic vector
    :param threshold: float, minimum distance at which
    :return: [float], positions of closest pass to atoms"""

    # Compute the displacement of this path
    traj_length = np.linalg.norm(vector)

    # Convert the start point to Cartesian coordinates
    start_pos = cell.conv_strc.lattice.get_cartesian_coords(start_pos)

    # Get the list of atoms that are likely to experience a 'near-hit'

    #   Determine the spacing to check along the line
    #     We will look for neighbors at several points along the line
    n_spacings = int(traj_length / threshold / np.sqrt(2)) + 1
    step_size = np.linalg.norm(vector) / (n_spacings - 1)

    #   Determine the radius to search for neighbors
    #     For an atom to be within `threshold` of the line, it must be within this radius of a sample point
    radius = np.sqrt((step_size / 2) ** 2 + threshold ** 2)

    #   Determine the points along the line where we will look for atoms
    points = [start_pos + x * vector for x in np.linspace(0, 1, n_spacings)]

    #    Look for atoms near those points
    sites = []
    for point in points:
        near_sites = cell.simulation_cell.get_sites_in_sphere(point, radius)
        sites.extend([x[0] for x in near_sites])

    #    Determine the distance and position along the line from each atom to the line between
    traj_direction = vector / traj_length
    near_impact = []
    for site in sites:
        from_line = (site.coords - start_pos) - np.dot(site.coords - start_pos, traj_direction) * traj_direction
        from_line = np.linalg.norm(from_line)
        if from_line < threshold:
            # Determine the displacement at which the projectile is closest to this atom
            position = np.dot(site.coords - start_pos, traj_direction)

            if estimate_extrema:
                # Determine the expected positions of the maxima
                #   We assume that the main driver of the stopping power is the 'ion-ion'
                #    repulsion. This force is proportional to 1/r^2*cos(theta) where r is the
                #    distance between the particle and the nearest atomic core, and
                #    theta is the angle between the direction of travel and line
                #    between the projectile and atom. It works out that this means the
                #    maximum force is +/-d/sqrt(2) from the position of closest transit, where
                #    d is the distance between the projectile's path and this atom
                special_points = np.multiply([-1, 1], from_line / np.sqrt(2)) + position

                coordinates = [x / traj_length for x in special_points]
            else:
                # Determine the fraction position along the path
                coordinates = [position / traj_length]

            for coordinate in coordinates:
                # Determine whether it is before or after the start of the trajectory
                if coordinate < 0 or coordinate > 1:
                    continue

                # Determine whether this point has already been added
                if len(near_impact) == 0 or np.abs(np.subtract(near_impact, coordinate)).min() > 1e-6:
                    near_impact.append(coordinate)
    return sorted(set(near_impact))

def _create_frame_generator(cell, start_pos, traj_vec, vmag):
    """Create a function that generates a snapshot of an projectile moving along a certain trajectory

    The function takes a float between 0 and 1 as an argument. A value of 0 returns the projectile at the starting
    point of the trajectory. A value of 1 returns the particle at the first point where the trajectory repeats.
    Values between 0-1 are linearly spaced between the two.

    :param cell: [Cell], Cell object defined in Cell.py
    :param start_pos: [float], starting point in conventional cell fractional coordinates
    :param traj_vec: [int], directional of travel in primitive cell coordinates
    :param vmag: [float], projectile velocity
    :return: function that returns position as a function of time float->([float]*3, [float]*3)"""

    # Compute the start point in cartesian coordinates
    start_pos = np.dot(cell.conv_strc.lattice.matrix, start_pos)

    # Compute the velocity vector
    velocity_vec = vmag * np.array(traj_vec, float) / np.linalg.norm(traj_vec)

    # Create the function
    def output(x):
        position = traj_vec * x + start_pos
        return position, velocity_vec
    return output


def _create_model_inputs(cell, featurizers, start_pos, traj_vec, vmag):
    """Create a function that computes the inputs to the force model at a certain point along a trajectory

    As in `_create_frame_generator`, the function takes a float between 0 and 1 as an argument. A value of 0
    returns the inputs for the force model at the starting point of the trajectory. A value of 1 returns the inputs at the first
    point where the trajectory repeats. Values between 0-1 are linearly spaced between the two.

    :param cell: [Cell], Cell object defined in Cell.py
    :param start_pos: [float], starting point in primitive cell fractional coordinates
    :param traj_vec: [int], directional of travel in conventional cell coordinates
    :param vmag: [float], projectile velocity
    :return: function float->float"""

    generator = _create_frame_generator(cell, start_pos, traj_vec, vmag)

    def output(x):
        # Get the structure
        frame = generator(x)
        # Get the inputs to the model
        return featurizers.featurize(*frame)
    return output

def _create_force_calculator(cell, model, featurizers, start_pos, traj_vec, vmag):
    """Create a function that computes the force acting on a projectile at a certain point along a trajectory

    As in `_create_frame_generator`, the function takes a float between 0 and 1 as an argument. A value of 0
    returns the force at the starting point of the trajectory. A value of 1 returns the force at the first
    point where the trajectory repeats. Values between 0-1 are linearly spaced between the two.

    :param cell: [Cell], Cell object defined in Cell.py
    :param start_pos: [float], starting point in conventional cell fractional coordinates
    :param traj_vec: [int], directional of travel in conventional cell coordinates
    :param vmag: [float], projectile velocity
    :return: function float->float"""

    generator = _create_model_inputs(cell, featurizers, start_pos, traj_vec, vmag)

    def output(x):
        # Evaluate the model
        inputs = generator(x)
        res = model.predict(np.array([inputs]), verbose = 0)
        return res if res.shape == () else res[0]
    return output

def _convert_coordinate(cell, start_pos, vdir, coordinate):
    """
    unify the coordinate to the contravariant coordinate of the conventional unit cell
    """
    if (coordinate.lower() == 'cartesian'):
        start_pos = cell.cartesian_to_conventional(start_pos)
        vdir = cell.cartesian_to_conventional(vdir)
        return start_pos, vdir

    elif (coordinate.lower() == 'supercell'):
        start_pos = cell.simulation_to_cartesian(start_pos)
        vdir = cell.simulation_to_cartesian(vdir)
        print('cartesian coordinate', start_pos)
        return _convert_coordinate(start_pos, vdir, 'cartesian')

    elif (coordinate.lower() == 'conventional'):
        return start_pos, vdir

    else: 
        raise ValueError("Invalid coordinate specified")


def compute_stopping_power(cell, model, featurizers, start_pos, vdir, vmag=1, *, coordinate = 'conventional', hit_threshold=2,
                           max_spacing=0.001, abserr=0.001, full_output=0, **kwargs):
    """Compute the stopping power along a trajectory.

    :param cell: [Cell], Cell object defined in Cell.py
    :param start_pos: [float], starting point in conventional cell fractional coordinates
    :param vdir: [int], directional of travel in conventional cell coordinates
    :param vmag: [float], magnitude of projectile velocity
    :param coordinate: [str], specify the coordinate of the start_pos and vdir. could be 'conventional': contravariant coordinate of the conventional unit cell; 'supercell': contravariant of the tddft supercell; 'cartesian': cartesian coordinates
    :param hit_threshold: float, threshold distance for marking when the trajectory passes close enough to an
            atom to mark the position of closest pass as a discontinuity to the integrator.
    :param abserr: [float], desired level of accuracy
    :param full_output: [0 or 1], whether to return the full output from `scipy.integrate.quad`
    :param kwargs: these get passed to `quad`"""

    start_pos, vdir = _convert_coordinate(cell, start_pos, vdir, coordinate)
    logging.info(f"Contravariant starting position in conventional unit cell {start_pos}")
    logging.info(f"Contravariant velocity direction in conventional unit cell {vdir}, velocity magnitude is {vmag} at. u.")

    # Create the integration function
    traj_vec = compute_trajectory(cell, vdir)
    f = _create_force_calculator(cell, model, featurizers, start_pos, traj_vec, vmag)

    # Determine the locations of peaks in the function (near hits)
    near_points = _find_near_hits(cell, start_pos, traj_vec, threshold=hit_threshold,
                                       estimate_extrema=True)
    
    # Determine the maximum number of intervals such that the maximum number of evaluations is below 
    #   a certain effective spacing
    traj_length = np.linalg.norm(traj_vec)
    max_inter = int(max(50, traj_length / max_spacing / 21)) # QUADPACK uses 21 points per interval

    # Perform the integration
    return quad(f, 0, 1, epsabs=abserr, full_output=full_output, points=near_points, limit=max_inter, **kwargs)

def create_force_calculator_given_displacement(cell, start_pos, traj_dir):
    """Create a function that computes the stopping force given displacement and current velocity
    
    :param start_pos: [float], starting point in conventional cell fractional coordinates
    :param traj_dir: [float], directional of travel in cartesian coordinate
    :return: (float, float)->float Takes displacement in distance units and velocity magnitude and computes force
    """
    
    # Get the trajectory direction as a unit vector
    traj_dir = np.divide(traj_dir, np.linalg.norm(traj_dir))
    
    # Convert the start point to Cartesian coordinates
    start_pos = cell.conv_strc.lattice.get_cartesian_coords(start_pos)
    print("start_pos", start_pos)
    
    # Make the function
    def output(disp, vel_mag, variance = np.array([0, 0, 0])):
        pos = start_pos + disp * traj_dir
        x = featurizers.featurize(pos + variance, vel_mag * traj_dir)
        return model.predict(np.array([x]), verbose = 0)[0].item()
    return output

if __name__ == '__main__':
    from ase.atoms import Atoms, Atom
    from stopping_power_ml.features import ProjectedAGNIFingerprints
    from sklearn.dummy import DummyRegressor

    # Create the example cell: A 4x4x4 supercell of fcc-Cu
    atoms = Atoms('Cu4', scaled_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0., 0.5], [0, 0.5, 0.5]],
                  cell=[3.52, ]*3, momenta=[[0, 0, 0], ]*4).repeat([4, 4, 4])
    atoms.append(Atom('H', [3.52/4, 3.52/4, 0], momentum=[1, 0, 0]))

    # Create the trajectory integrator
    featurizer = ProjectedAGNIFingerprints(atoms, etas=None)
    model = DummyRegressor().fit([[0,]*8], [1])
    tint = TrajectoryIntegrator(atoms, model, [featurizer])

    # Make sure it gets the correct trajectory distance for a [1 1 0] conventional cell lattice vector.
    #  This travels along the face of the FCC conventional cell, and repeats halfway across the face.
    #  So, the minimum trajectory is [0.5, 0.5, 0] in conventional cell coordinates
    assert np.isclose(tint.compute_trajectory([1, 1, 0]), [1.76, 1.76, 0]).all()

    # Another test: [2 0 0]. This trajectory repeats after [1, 0, 0]
    assert np.isclose(tint.compute_trajectory([2, 0, 0]), [3.52, 0, 0]).all()

    # Make sure the frame generator works properly
    f = tint._create_frame_generator([0, 0, 0], [1, 0, 0], 1)
    pos, vel = f(1)
    assert np.isclose([3.52, 0, 0], pos).all()
    assert np.isclose([1, 0, 0], vel).all()

    f = tint._create_frame_generator([0.25, 0.25, 0.25], [1, 1, 1], np.sqrt(3))
    pos, vel = f(0.5)
    assert np.isclose([3.52 * 0.75,]*3, pos).all()
    assert np.isclose([1, 1, 1], vel).all()

    # Test the force generator (make sure it does not crash)
    f = tint._create_force_calculator([0.25, 0, 0], [1, 0, 0], 1)
    assert np.isclose(f(0), 1)  # The model should produce 1 for all inputs

    # Test the integrator
    result = tint.compute_stopping_power([0.25, 0, 0], [1, 0, 0], 1)
    assert np.isclose(result[0], 1)

    # Find near impacts
    result = tint._find_near_hits([0, 0, 0], [1, 0, 0], threshold=1)
    assert len(result) == 2
    assert np.isclose(result, [0, 1]).all()

    result = tint._find_near_hits([0, 0, 0], [1, 0, 0], threshold=2)
    assert len(result) == 3
    assert np.isclose(result, [0, 0.5, 1]).all()

    result = tint._find_near_hits([0.2, 0, 0], [1, 0, 0], threshold=2)
    assert len(result) == 2
    assert np.isclose(result, [0.3, 0.8]).all()

    result = tint._find_near_hits([0, 0, 0.4], [1, 0, 0], threshold=1)
    assert len(result) == 1
    assert np.isclose(result, [0.5]).all()

    assert np.isclose(tint._find_near_hits([0,0.75,0.85], [5,-1,-1], 0.5), [0.8])

    # Find peaks around near impacts
    result = tint._find_near_hits([0, 0, 0], [1, 0, 0], threshold=1, estimate_extrema=True)
    assert len(result) == 2
    assert np.isclose(result, [0, 1]).all()

    result = tint._find_near_hits([0, 0, 0], [1, 0, 0], threshold=2, estimate_extrema=True)
    assert len(result) == 4
    assert np.isclose(result, [0, 0.5 * (1 - np.sqrt(0.5)), 0.5 * (1 + np.sqrt(0.5)), 1]).all()

    result = tint._find_near_hits([0, 0, 0.4], [1, 0, 0], threshold=1, estimate_extrema=True)
    delta = 0.1 / np.sqrt(2)
    assert len(result) == 2
    assert np.isclose(result, [0.5 - delta, 0.5 + delta]).all()
