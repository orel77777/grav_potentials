import math
from scipy.integrate import quad
import mpmath

def norm_round_thor_potential_at_x3_eq_zero(theta,r,x3,r0,R0):
	R1 = (1+((r0/R0)*math.cos(theta)))
	a = (2*((r**2)+((x3-(r0*math.sin(theta)))**2))/(R0**2))
	b = (((a/2)-(R1**2))+math.sqrt((((a/2)+(R1**2))**2)-((4*(R1**2))*(((r)**2)/(R0**2)))))
	c = (((a/2)-(R1**2))-math.sqrt((((a/2)+(R1**2))**2)-((4*(R1**2))*(((r)**2)/(R0**2)))))
	n = ((a-b)/(2*((r**2)/(R0**2))))
	k_up = math.sqrt(((a/2)+(R1**2)-(2*R1*r/R0))/((a/2)+(R1**2)+(2*R1*r/R0)))
	k1 = ((1-k_up)/(1+k_up))
	phi_norm = (((math.cos(theta))/(math.sqrt(a-c)))*(((c+(2*((R1**2)-(((r**2)/(R0**2))))))*((mpmath.ellipk(k1**2))))+((a-c)*((mpmath.ellipe(k1**2))))-((a-(((2*(r**2))/(R0**2))))*((mpmath.ellippi(n,k1**2))))))
	return phi_norm
