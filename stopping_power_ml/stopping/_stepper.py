#!/usr/bin/env python
# coding=utf-8
import numpy as np

def _velocity_verlet(position, velocity, acceleration, force_calc, mass, dt):

    velocity_half = velocity + 0.5 * acceleration * dt

    position_next = position + velocity_half * dt

    f = force_calc(position_next, velocity_half)

    acceleration_next = -f/mass

    velocity_next = velocity_half + 0.5 * acceleration_next * dt

    return position_next, velocity_next, acceleration_next

def _rk4(position, velocity, acceleration, force_calc, mass, dt):
    """ Runge-Kutta 4th order integration scheme """
    k1v = -force_calc(position, velocity)/mass
    k1x = velocity

    k2v = -force_calc(position + 0.5 * k1x * dt, velocity + 0.5 * k1v * dt)/mass
    k2x = velocity + 0.5 * k1v * dt
    
    k3v = -force_calc(position + 0.5 * k2x * dt, velocity + 0.5 * k2v * dt)/mass
    k3x = velocity + 0.5 * k2v * dt
    
    k4v = -force_calc(position + k3x * dt, velocity + k3v * dt)/mass
    k4x = velocity + k3v * dt

    velocity_next = velocity + (1/6) * (k1v + 2*k2v + 2*k3v + k4v) * dt
    position_next = position + (1/6) * (k1x + 2*k2x + 2*k3x + k4x) * dt

    return position_next, velocity_next, k4v
