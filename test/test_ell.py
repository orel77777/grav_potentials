#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 02:05:08 2020

@author: ivan
"""
import potentials
import pytest
import numpy as np
import math


@pytest.mark.parametrize('a', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_ellipt_ball(a):
    ell = potentials.norm_ellipsoid_potential(a, a, a, a, a, a)
    ball = potentials.norm_ball_potential(math.sqrt(3*(a**2)), a)
    np.testing.assert_allclose(ell, ball, atol=1e-5)


@pytest.mark.parametrize('a,r', [(1, 1e6),
                                 (1, 1e7),
                                 (1, 1e8),
                                 (1, 1e9),
                                 (1, 1e10)])
def test_ellipt_asimpt(a, r):
    ell = potentials.norm_ellipsoid_potential(r, r, r, a, a, a)
    ball = potentials.norm_ball_potential(math.sqrt(3*(r**2)), a)
    np.testing.assert_allclose(ell, ball, atol=1e-5)


@pytest.mark.parametrize('R,r', [(1, 1e6),
                                 (1, 1e7),
                                 (1, 1e8),
                                 (1, 1e9),
                                 (1, 1e10)])
def test_round_disk_and_ring_asimpt_r(R, r):
    disk = potentials.norm_round_disk_potential(r, 0, R)
    ring = potentials.norm_round_ring_potential(r, 0, R)
    np.testing.assert_allclose(disk, ring, atol=1e-5)


@pytest.mark.parametrize('R,x3', [(1, 1e6),
                                  (1, 1e7),
                                  (1, 1e8),
                                  (1, 1e9),
                                  (1, 1e10)])
def test_round_disk_and_ring_asimpt_x3(R, x3):
    disk = potentials.norm_round_disk_potential(0, x3, R)
    ring = potentials.norm_round_ring_potential(0, x3, R)
    np.testing.assert_allclose(disk, ring, atol=1e-5)


@pytest.mark.parametrize('a1, a2, a3', [(1, 1, 1),
                                        (2, 2, 2),
                                        (3, 3, 3),
                                        (4, 4, 4),
                                        (5, 5, 5)])
def test_cube_inner_center(a1, a2, a3):
    L = math.sqrt((a1**2)+(a2**2)+(a3**2))
    Q = math.atan((a1*a3)/((a2**2)+(a3**2)+(a2*L)))
    P = math.atan((a1*a3)/((a2**2)+(a3**2)-(a2*L)))
    theor_center_pot = ((a2*a3*math.log((L+a1)/(L-a1))) +
                        (a1*a3*math.log((L+a2)/(L-a2))) +
                        (a1*a2*math.log((L+a3)/(L-a3))) +
                        (0.5*(a3**2)*(Q-P)) +
                        (0.5*(a2**2)*(Q-P)) +
                        (0.5*(a1**2)*(Q-P)))
    lib_center_pot = potentials.norm_inner_cube_potential(0, 0, 0, a1, a2, a3)
    np.testing.assert_allclose(theor_center_pot, lib_center_pot, atol=1e-5)


@pytest.mark.parametrize('a1, a2, a3', [(1, 1, 1),
                                        (2, 2, 2),
                                        (3, 3, 3),
                                        (4, 4, 4),
                                        (5, 5, 5)])
def test_cube_rel_center_and_top(a1, a2, a3):
    eps = 1e-7
    center_pot = potentials.norm_inner_cube_potential(0, 0, 0, a1, a2, a3)
    vertex_pot = potentials.norm_inner_cube_potential(((a1*0.5)-eps),
                                                      ((a2*0.5)-eps),
                                                      ((a3*0.5)-eps),
                                                      a1, a2, a3)
    rel = (center_pot/vertex_pot)
    np.testing.assert_allclose(rel, 2, atol=1e-5)


@pytest.mark.parametrize('x2, x3, R, H', [(0.5, 0.5, 1, 2),
                                          (0.5, 0.5, 2, 50),
                                          (0.5, 0.5, 3, 75),
                                          (0.5, 0.5, 4, 100),
                                          (0.5, 0.5, 5, 125)])
def test_cyl_inner_round_pot(x2, x3, R, H):
    r = math.sqrt((x2**2)+(x3**2))
    round_pot_inner = math.pi * \
        ((2*(R**2)*math.log(2*math.sqrt(math.exp(1))*H/R))-(r**2))
    ell_pot = potentials.norm_ell_cyl_potential(x2, x3, H, R, R)
    np.testing.assert_allclose(round_pot_inner, ell_pot, atol=1e-5)
