#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:47:47 2020

@author: ivan
"""
import matplotlib.pyplot as plt
import numpy as np
import potentials
import argparse

def_step_r_u = 0.1
def_step_r_l = 0.01

def_step_q_l = 0.1
def_step_q_u = 0.2

q_initial = 0.1
q_last = 0.5

r_R0_up = 2
r0 = 1


def parse_args():
    pars = argparse.ArgumentParser()
    pars.add_argument("-r", "--r",
                      help='step r: 0.01 -- 0.1 (Default value = 0.1)',
                      default=def_step_r_u, type=float)
    pars.add_argument("-q", "--q",
                      help='step q: 0.1  -- 0.2 (Default value = 0.1)',
                      default=def_step_q_l, type=float)
    args = pars.parse_args()
    if((args.r > def_step_r_u) or (args.r < def_step_r_l)):
        args.r = def_step_r_u
    if((args.q > def_step_q_u) or (args.q < def_step_q_l)):
        args.q = def_step_q_l
    return {'r': args.r, 'q': args.q}


def main():
    steps = parse_args()
    step_q = steps['q']
    step = steps['r']

    lq = np.arange(q_initial, q_last+step_q, step_q)
    R0 = r0/lq

    r_R0 = np.arange(step, r_R0_up, step)
    r = r_R0[:, np.newaxis]*R0

    R0 = np.rot90(np.repeat(R0, r.shape[0]).reshape(R0.shape[0], r.shape[0]))
    phi_norm = np.vectorize(potentials.norm_round_thor_potential)(r, 0, r0, R0)

    rR0max = r_R0[np.argmax(phi_norm, 0)]
    phimax = np.amax(phi_norm, 0)

    z = plt.plot(r_R0, phi_norm, '.-')
    plt.grid(True)
    plt.xlabel(r'$\frac{r}{R_0}$', fontsize=17)
    plt.ylabel(r'$\frac{\varphi(r)}{\frac{8}{3} \pi G \rho R_0 r_0 }$',
               fontsize=17)
    labels = np.core.defchararray.add(np.repeat(r'$q = \frac{r_0}{R_0} = $', lq.shape[0]),
                                      np.round(lq, 3).astype(str))
    plt.legend(iter(z), labels)
    plt.tight_layout()
    plt.show()

    plt.plot(lq, rR0max, '.-')
    plt.grid(True)
    plt.xlabel(r'q = $\frac{r_0}{R_0}$', fontsize=17)
    plt.ylabel(r'$(\frac{r}{R_0})_{max}$', fontsize=17)
    plt.tight_layout()
    plt.show()

    plt.plot(lq, phimax, '.-')
    plt.grid(True)
    plt.xlabel(r'q = $\frac{r_0}{R_0}$', fontsize=17)
    plt.ylabel(r'$( \frac{\varphi(r)}{\frac{8}{3} \pi G \rho R_0 r_0 } )_{max}$',
               fontsize=17)
    plt.tight_layout()
    plt.show()


