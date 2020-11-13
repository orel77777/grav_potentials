#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 02:05:08 2020

@author: ivan
"""
import potentials
import pytest
import math


def test_ellipt_ball():
	assert (abs(potentials.norm_ellipsoid_potential(3,3,3,3,3,3)-potentials.norm_ball_potential(math.sqrt(3**3),3)) < 1e-6)