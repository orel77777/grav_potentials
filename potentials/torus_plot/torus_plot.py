#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:47:47 2020

@author: ivan
"""
import matplotlib.pyplot as plt
import potentials

def main():
	step = 0.1
	step_q = 0.2
	q_initial = 0.1
	q_last = 0.9
	r_min = 1e-10

	x3 = 0
	qr_initial_up = 2
	r0 = 1

	lq =[]
	il = q_initial
	while(il < q_last):
		lq.append(il)
		il+=step_q

	lrR0max = []
	lphimax = []
	for q in lq:
	
		print(q)
		R0 = r0/q
	
		r_initial_up = R0*qr_initial_up
	
		lphi_norm = []
		lr = []
		r = r_initial_up
		while(r > r_min):
			lphi_norm.append(potentials.norm_round_thor_potential(r,0,r0,R0))
			lr.append(r)	
			r=r-step
	
		for i in enumerate(lr):
			lr[i[0]] = lr[i[0]]/R0
	
		max_lphi_norm = max(lphi_norm)
		lphimax.append(max_lphi_norm)
		lrR0max.append(lr[lphi_norm.index(max_lphi_norm)])	
	
		plt.plot(lr, lphi_norm, '.-', label=r'$q = \frac{r_0}{R_0} =$' + str(round(q,2)))
		plt.grid(True)
		plt.xlabel(r'$\frac{r}{R_0}$', fontsize=17)
		plt.ylabel(r'$\frac{\varphi(r)}{\frac{8}{3} \pi G \rho R_0 r_0 }$', fontsize=17)
		plt.legend()
	plt.tight_layout()
	plt.show()
	
	plt.plot(lq, lrR0max, '.-')
	plt.grid(True)
	plt.xlabel(r'q = $\frac{r_0}{R_0}$', fontsize=17)
	plt.ylabel(r'$(\frac{r}{R_0})_{max}$', fontsize=17)
	plt.tight_layout()
	plt.show()

	plt.plot(lq, lphimax, '.-')
	plt.grid(True)
	plt.xlabel(r'q = $\frac{r_0}{R_0}$', fontsize=17)
	plt.ylabel(r'$( \frac{\varphi(r)}{\frac{8}{3} \pi G \rho R_0 r_0 } )_{max}$', fontsize=17)
	plt.tight_layout()
	plt.show()
