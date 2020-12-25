import math
from scipy.integrate import quad
import mpmath
from sympy.solvers import solve
from sympy import Symbol
import numpy as np

__all__ = ('norm_round_thor_potential',
	'norm_round_disk_potential',
	'norm_round_ring_potential',
	'norm_ellipsoid_potential',
	'norm_ball_potential',
	'norm_inner_cube_potential',
	'norm_ell_cyl_potential')

def diff_norm_round_thor_potential(theta, r, x3, r0, R0):
	R1 = (1+((r0/R0)*math.cos(theta)))
	a = (2*((r**2)+((x3-(r0*math.sin(theta)))**2))/(R0**2))
	b = (((a/2)-(R1**2))+math.sqrt((((a/2)+(R1**2))**2)-((4*(R1**2))*(((r)**2)/(R0**2)))))
	c = (((a/2)-(R1**2))-math.sqrt((((a/2)+(R1**2))**2)-((4*(R1**2))*(((r)**2)/(R0**2)))))
	n = ((a-b)/(2*((r**2)/(R0**2))))
	k_up = math.sqrt(((a/2)+(R1**2)-(2*R1*r/R0))/((a/2)+(R1**2)+(2*R1*r/R0)))
	k1 = ((1-k_up)/(1+k_up))
	phi_norm = (((math.cos(theta))/(math.sqrt(a-c)))*(((c+(2*((R1**2)-(((r**2)/(R0**2))))))*((mpmath.ellipk(k1**2))))+((a-c)*((mpmath.ellipe(k1**2))))-((a-(((2*(r**2))/(R0**2))))*((mpmath.ellippi(n,k1**2))))))
	return phi_norm

def norm_round_thor_potential(r, x3, r0, R0, precise_border=1e-16):
	'''
	Normalized potential of a homogeneous torus to 8*G*\rho*\pi*R0*r0/3 at the point with cylindrical coordinates (r, \phi, x3).
	Where G is the gravitational constant and \rho is the density. 
	The torus has angular symmetry in \phi, so there is only two input parameter - distance r and z coordinate.
	The center of the torus is considered to be at the origin.

	Parameters
	----------
	r : real
		r coordinate in a cylindrical coordinate system.
	x3 : real
		z coordinate in cartesian system.
	r0 : real
		r0 is distance to the center of a circle in a torus section.
	R0 : real
		R0 is the radius of the circle in the section of the torus.
	precise_border : TYPE, optional
		In order to bypass the singular points during the numerical integration in the formulas for the torus potential,
		it is necessary to deviate from them by a certain number - <presize border>.
		In fact, the definition of a function by continuity. The default value is 1e-16.

	Returns
	-------
	real
		normilized torus potential at (r, x3).

	'''
	return (quad(diff_norm_round_thor_potential, precise_border, (math.pi-precise_border), args=(r,x3,r0,R0))[0]
					+quad(diff_norm_round_thor_potential, (math.pi+precise_border), ((2*math.pi)-precise_border), args=(r,x3,r0,R0))[0])*(3/(2*math.pi*math.sqrt(2)))
	
def norm_round_ring_potential(r, x3, R):
	'''
	Normalized potential of a homogeneous ring to G\sigma at the point with cylindrical coordinates (r, \phi, x3).
	Where G is the gravitational constant and \rho is the density. 
	The ring has angular symmetry in \phi, so there is only two input parameter - distance r and z coordinate.
	The center of the ring is considered to be at the origin.

	Parameters
	----------
	r : real
		r coordinate in a cylindrical coordinate system.
	x3 : real
		z coordinate in cartesian system.
	R : real
		ring radius.

	Returns
	-------
	phi : real
		normilized ring potential at (r, x3).

	'''
	phi = ((4*R)/(math.sqrt(((R+r)**2)+(x3**2))))*mpmath.ellipk(((4*R*r)/(((R+r)**2)+(x3**2))))
	return float(phi)

def norm_round_disk_potential(r, x3, R):
	'''
	Normalized potential of a homogeneous solid disk to G\sigma at the point with cylindrical coordinates (r, \phi, x3).
	Where G is the gravitational constant and \sigma is the density. 
	The solid disk has angular symmetry in \phi, so there is only one input parameter - distance r and z coordinate.
	The center of the disk is considered to be at the origin.
	
	Parameters
	----------
	r : real
		r coordinate in a cylindrical coordinate system.
	x3 : real
		z coordinate in cartesian system.
	R : real
		disk radius.

	Returns
	-------
	phi_norm : real
		normilized disk potential at (r, x3).

	'''
	if((r==0) and (x3!=0)):
		return float((2*math.pi*((math.sqrt((R**2)+(x3**2))-abs(x3)))))	
	elif((x3==0) and (r!=0)):
		if(r>R):
			return float((4/r)*(((r**2)*mpmath.ellipe((R/r)**2))-(((r**2)-(R**2))*mpmath.ellipk((R/r)**2))))
		elif(r<=R):
			return float(((4*R)*(mpmath.ellipe((r/R)**2))))
	elif((x3!=0) and (r!=0)):
		alpha_quad = (R+math.sqrt((r**2)+(x3**2)))**2
		beta_quad  = (R-math.sqrt((r**2)+(x3**2)))**2
		a = math.sqrt(((R+r)**2)+(x3**2))
		b = math.sqrt(((R-r)**2)+(x3**2))
		k = 2*math.sqrt(R*r)/a
		n1 = (alpha_quad*((a**2)-(b**2)))/((a**2)*((alpha_quad)-(b**2)))
		n2 = (beta_quad*((a**2)-(b**2)))/((a**2)*((beta_quad)-(b**2)))
		phi_norm = 4*((-math.pi*0.5*x3) + ((((R**2)-(r**2)-(x3**2))/(2*a))*(1+((4*(R**2)*(x3**2))/(alpha_quad*beta_quad)))*mpmath.ellipk(k**2))+(((x3**2)*(R*(b**2)/a))*(((mpmath.ellippi(n1,k**2))/((alpha_quad-(b**2))*(R+math.sqrt((r**2)+(x3**2)))))+((mpmath.ellippi(n2,k**2))/((beta_quad-(b**2))*(R-math.sqrt((r**2)+(x3**2)))))))+(a*0.5*mpmath.ellipe(k**2)))
		return float(phi_norm)
	
def norm_ellipsoid_potential(x1, x2, x3, a1, a2, a3):
	'''
	Normalized potential of a homogeneous triaxial ellipsoid to G\rho at the point with cartesian coordinates (x1, x2, x3).
	Where G is the gravitational constant and \rho is the density. 
	The center of the ellipsoid is considered to be at the origin.
	The axes of coordinate system are aligned with the principal axes of the ellipsoid.

	Parameters
	----------
	x1 : real
		x coordinate in cartesian system.
	x2 : real
		y coordinate in cartesian system.
	x3 : real
		z coordinate in cartesian system.
	a1 : real
		a1 - a - first semi-major axis of the ellipsoid.
	a2 : real
		a2 - b - second semi-major axis of the ellipsoid.
	a3 : real
		a3 - c - third semi-major axis of the ellipsoid.

	Returns
	-------
	phi_res : real
		normilized ellipsoid potential at (x1, x2, x3).

	'''
	def delta(vv):
		return math.sqrt(((a1**2)+vv)*((a2**2)+vv)*((a3**2)+vv))
	def integral(v):
		return (1/(delta(v)))*(1-((x1**2)/((a1**2)+v))-((x2**2)/((a2**2)+v))-((x3**2)/((a3**2)+v)))
	if((((x1/a1)**2)+((x2/a2)**2)+((x3/a3)**2))>1):
		lambda_c = Symbol('lambda_c', real=True)
		lambda_ccl = solve(((((x1**2)/((a1**2)+lambda_c))+((x2**2)/((a2**2)+lambda_c))+((x3**2)/((a3**2)+lambda_c)))-1), lambda_c)

		if(lambda_ccl[0]>=0):
			lambda_cc = lambda_ccl[0]
		elif(lambda_ccl[1]>=0):
			lambda_cc = lambda_ccl[1]
		elif(lambda_ccl[2]>=0):
			lambda_cc = lambda_ccl[2]

		phi_res = math.pi*a1*a2*a3*quad(integral, lambda_cc, np.inf)[0]
	else:
		phi_res = math.pi*a1*a2*a3*quad(integral, 0, np.inf)[0]
	return phi_res

def norm_ball_potential(r, R):
	'''
	Normalized potential of a homogeneous ball to G\rho at the point with spherical coordinates (r, \phi, \theta).
	Where G is the gravitational constant and \rho is the density. 
	The ball has angular symmetry, so there is only one input parameter - distance r.
	The center of the ball is considered to be at the origin.
	
	Parameters
	----------
	r : real
		distance to point.
	R : real
		ball radius.

	Returns
	-------
	phi_norm : real
		normilized ball potential at r.

	'''
	if(r<=R):	
		phi_norm = (2/3)*math.pi*((3*(R**2))-(r**2))
	elif(r>R):
		phi_norm = ((4*math.pi*(R**3))/(3*r))
	return phi_norm

def norm_inner_cube_potential(x1, x2, x3, a1, a2, a3):
	'''
	Normalized inner potential of a homogeneous parallelepiped to G\rho at the point with cartesian coordinates (x1, x2, x3).
	Where G is the gravitational constant and \rho is the density. 
	The origin is considered at the intersection of the parallelepiped diagonals.
	
	Parameters
	----------
	x1 : real
		x coordinate in cartesian system.
	x2 : real
		y coordinate in cartesian system.
	x3 : real
		z coordinate in cartesian system.
	a1 : real
		a1 - a - first axis of the parallelepiped.
	a2 : real
		a2 - b - second axis of the parallelepiped.
	a3 : real
		a3 - c - third axis of the parallelepiped.

	Returns
	-------
	phi_res : real
		normilized inner parallelepiped potential at (x1, x2, x3).

	'''
	ksi1 = (x1-(a1*0.5))
	ksi2 = (x1+(a1*0.5))
	
	eta1 = (x2-(a2*0.5))
	eta2 = (x2+(a2*0.5))
	
	dzeta1 = (x3-(a3*0.5))
	dzeta2 = (x3+(a3*0.5))
	
	
	def ph_abcd(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2):
		ra = math.sqrt((ksi1**2)+(eta2**2)+(dzeta2**2))
		rb = math.sqrt((ksi1**2)+(eta1**2)+(dzeta2**2))
		rc = math.sqrt((ksi2**2)+(eta1**2)+(dzeta2**2))
		rd = math.sqrt((ksi2**2)+(eta2**2)+(dzeta2**2))
		return dzeta2*((ksi1*math.log((eta1+rb)/(eta2+ra)))-(ksi2*math.log((eta1+rc)/(eta2+rd)))-(eta1*math.log((((rb-ksi1)*(rc+ksi2))/((eta1**2)+(dzeta2**2)))))+(eta2*math.log((((ra-ksi1)*(rd+ksi2))/((eta2**2)+(dzeta2**2)))))+(dzeta2*(math.atan((ksi1*dzeta2)/((eta1**2)+(eta1*rb)+(dzeta2**2)))-math.atan((ksi2*dzeta2)/((eta1**2)+(eta1*rc)+(dzeta2**2)))))-(dzeta2*(math.atan((ksi1*dzeta2)/((eta2**2)+(eta2*ra)+(dzeta2**2)))-math.atan((ksi2*dzeta2)/((eta2**2)+(eta2*rd)+(dzeta2**2))))))
	def ph_efgh(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2):
		return ph_abcd(ksi1, ksi2, eta1, eta2, dzeta1, (-dzeta1))
	def ph_bcgf(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2):
		return ph_efgh(ksi1, ksi2, dzeta1, dzeta2, eta1, eta2)
	def ph_abfe(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2):
		return ph_efgh(dzeta1, dzeta2, eta1, eta2, ksi1, ksi2)
	def ph_adhe(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2):
		return ph_bcgf(ksi1, ksi2, -eta2, eta2, dzeta1, dzeta2)
	def ph_dcgh(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2):
		return ph_abfe(-ksi2, ksi2, eta1, eta2, dzeta1, dzeta2)
	
	phi_norm = (0.5*(ph_abcd(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2)+ph_efgh(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2)+ph_abfe(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2)+ph_dcgh(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2)+ph_bcgf(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2)+ph_adhe(ksi1, ksi2, eta1, eta2, dzeta1, dzeta2)))
	return phi_norm


def norm_ell_cyl_potential(x2, x3, H, a2, a3):
	'''
	Normalized potential of a homogeneous cylinder with an elliptical section to G\rho at the point with cartesian coordinates (x2, x3).
	Where G is the gravitational constant and \rho is the density.
	H>>max(a2, a3).
	The origin is considered at the center of the cylinder.
	The cylinder is elongated along the x1 axis and 2*H is the height of the cylinder along the x1 axis.
	
	Parameters
	----------
	
	x2 : real
		x cartesian coordinate in the plane of symmetry 
	x3 : real
		y cartesian coordinate in the plane of symmetry
	H : real
		H - is the height along x1 axis.
	a2 : real
		a2 - a - first semi-major axis of the ellipse in the plane of symmetry.
	a3 : real
		a3 - b - second semi-major axis of the ellipse  in the plane of symmetry.

	Returns
	-------
	phi_res : real
		normilized cylinder potential with an elliptical section at (x2, x3).

	'''
	e = math.exp(1)
	if((((x2**2)/(a2**2))+((x3**2)/(a3**2))) > 1):
		alpha = (((a2**2)-(a3**2))/4)
		r = math.sqrt((x2**2)+(x3**2))
		a = (1-(4*alpha*(((x2**2)-(x3**2))/(r**4))))
		b = (8*alpha*abs(x2*x3)/(r**4))
		R = math.sqrt((a**2)+(b**2))
		K = math.sqrt((R+a)/2)
		L = math.sqrt((R-a)/2)
		phi_norm = (2*math.pi*a2*a3*(math.log((4*H*math.sqrt(e))/(math.sqrt((a2**2)-(a3**2))))+(0.25*math.log((((K-1)**2)+(L**2))/(((K+1)**2)+(L**2))))+((((K-1)*((x2**2)-(x3**2)))-(2*L*abs(x2*x3)))/((a2**2)-(a3**2)))))
	else:
		A2 = ((2*a3)/(a2+a3))
		A3 = ((2*a2)/(a2+a3))
		phi_norm = (math.pi*((2*a2*a3*math.log((4*math.sqrt(e)*H)/(a2+a3)))-(A2*(x2**2))-(A3*(x3**2))))
	
	return phi_norm
