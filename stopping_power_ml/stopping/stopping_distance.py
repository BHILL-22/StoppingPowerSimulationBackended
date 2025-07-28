"""Tools for computing the stopping distance of a projectile"""

from scipy.optimize import minimize_scalar
from scipy.integrate import RK45, LSODA, DOP853, Radau, BDF
from time import perf_counter, time
from copy import copy
import pandas as pd
import numpy as np
import tempfile
import keras
from ._stepper import _velocity_verlet, _rk4

import functools
print = functools.partial(print, flush=True)

class StoppingDistanceComputer:
    """Utility tool used compute the stopping distance"""
    
    def __init__(self, traj_int, *, proj_mass=1837, max_step = 50, rtol = 1e-5, atol = 1e-7, stepper = 'rk45'):
        """Initialize the stopping distance computer
        
        Args:
            traj_int (TrajectoryIntegrator): Tool used to create the force calculator
            proj_mass (float): Mass of the projectile in atomic units
            max_step (float): Maximum timestep size allowed by the integrator
        """
        
        self.traj_int = traj_int
        self.proj_mass = proj_mass
        self.max_step = max_step
        self.rtol = rtol
        self.atol = atol
        self.stepper = stepper.lower()

    def __getstate__(self):
        # Save the model embedded in the TrajInt 
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self.traj_int.model, fd.name, overwrite=True)
            model_str = fd.read()
        
        # Assemble the pickle object
        d = self.__dict__.copy()
        d.update({ 'model_str': model_str })
        
        # Modify the traj_int to have model be a placeholder
        d['traj_int'] = copy(self.traj_int)  # So as to not effect self
        d['traj_int'].model = 'placeholder'
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        
        # Load the Keras model from disk 
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        
        # Add it to the traj_int
        self.traj_int.model = model
        
    def _make_ode_function(self, start_point, start_traj):
        """Make the function used to run the ODE

        Args:
            start_point ([float]*3): Starting point of the run
            start_traj ([float]*3): Starting direction
        """

        # Make the force calculator
        force_calc = self.traj_int.create_force_calculator_given_displacement(start_point, start_traj)
        
        def output(t, y):
            # Get the velocity and displacement
            v, x = y

            # Compute the force
            ts = time()
            f = force_calc(x, v)
            return [-f / self.proj_mass, v]
        return output


    def _in_house_stepper(self, start_point, start_velocity, stop_velocity_mag, max_time, output, status, stepper):
        start_time = time()

        force_calc = self.traj_int.create_force_calculator_given_displacement(start_point, start_velocity)
        v = np.linalg.norm(start_velocity)
        x = 0
        acc = 0

        dt = self.max_step
        t = 0

        i = 0
        states = [(0, v, 0, time() - start_time)]

        while v > stop_velocity_mag and t < max_time:
            wts = time()

            x, v, acc = stepper(x, v, acc, force_calc, self.proj_mass, dt)

            i += 1
            t += dt
            if output is not None and i % output == 0:
                states.append([t, v, x, time() - start_time])
                if status:
                    self._viewable(i, t, v, x, dt, time() - wts)
                
        stop_dist = x

        # Return the results
        if output is not None:
            return stop_dist, self._output(i, t, v, x, time() - start_time, states)
        return stop_dist

    def _scipy_stepper(self, start_point, start_velocity, stop_velocity_mag, max_time, output, status, stepper):
        """Compute the stopping distance of a projectile
        
        Args:
            start_point ([float]*3): Starting point of the run. In fractional coordinates of conventional cell
            start_velocity ([float]*3): Starting velocity
            stop_velocity_mag (float): Velocity at which to stop the calculation
            max_time (float): Time at which to stop the solver (assuming an error)
            output (int): Number of timesteps between outputting status information
            status (bool): Whether to print status information to screen
        Returns:
            - (float) Stopping distance
            - (pd.DataFrame) Velocity as a function of position and time
        """
        start_time = time()

        # Make the force calculator
        fun = self._make_ode_function(start_point, start_velocity)
        
        # Compute the initial velocity
        v_init = np.linalg.norm(start_velocity)
        
        # Create the ODE solvers
        stepper = stepper(fun, 0, [v_init, 0], max_time, rtol = self.rtol, atol = self.atol, max_step = self.max_step)
        
        # Iterate until velocity slows down enough
        i = 0
        states = [(0, v_init, 0, time() - start_time)]

        while stepper.y[0] > stop_velocity_mag and stepper.t < max_time:
            wts = time()

            stepper.step()
            i += 1
            if output is not None and i % output == 0:
                states.append([stepper.t, *stepper.y, time() - start_time])
                if status:
                    stepper.t_old = 0 if i == 1 else stepper.t_old  
                    self._viewable(i, stepper.t, stepper.y[0], stepper.y[1], stepper.t - stepper.t_old, time() - wts)

        # Determine the point at which the velocity crosses the threshold
        #   ODE solvers give you an interpolator over the last timestep
        interp = stepper.dense_output()
        res = minimize_scalar(lambda x: np.abs(interp(x)[0] - stop_velocity_mag), bounds=(stepper.t_old, stepper.t))
        stop_dist = interp(res.x)[1]
                
        # Return the results
        if output is not None:
            return stop_dist, self._output(i, stepper.t, stepper.y[0], stepper.y[1], time() - start_time, states)
        return stop_dist

    def _output(self, step, t, v, x, ctime, states):
        print('\nStep: {} - Time: {} - Velocity: {} - Position: {} - total wall time {}'.format(step, t, v, x, ctime))
        states = pd.DataFrame(dict(zip(['time', 'velocity', 'displacement', 'sim_time'], np.transpose(states))))
        return states
    
    def _viewable(self, i, t, v, x, dt, wtime):
        print(f'Step: {i} - Time: {t:0.8f} - Velocity: {v} - Position: {x} - time step: {dt} - wall time: {wtime:0.4f} sec ', end = "\r")

    def compute_stopping_distance(self, start_point, start_velocity, *, stop_velocity_mag = 0.4, max_time = 3e5, output = None, status = True):
        scipy_steppers = {'rk45': RK45, 'lsoda': LSODA, 'dop853': DOP853, 'radau': Radau, 'bdf': BDF}
        in_house_steppers = {'rk4': _rk4, 'velocity_verlet': _velocity_verlet}
        if (self.stepper in scipy_steppers):
            return self._scipy_stepper(start_point, start_velocity, stop_velocity_mag, max_time, output, status, scipy_steppers[self.stepper])
        elif (self.stepper in in_house_steppers):
            return  self._in_house_stepper(start_point, start_velocity, stop_velocity_mag, max_time, output, status, in_house_steppers[self.stepper])
        else:
            raise NameError(f"{self.stepper} is not supported")
