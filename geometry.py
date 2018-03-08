import numpy as np
import numpy.core.numeric as NX
from numpy.linalg import norm
from numpy.lib.function_base import trim_zeros
from numpy.polynomial.polynomial import polyval2d
#import numpy.lib.polynomial._raise_power as _raise_power

def rayTrace():
    p = np.array((0, .25))
    x = np.arange(-.002, .0021, .0001)
    xy = np.pad(np.reshape(x,(-1,1)),((0,0),(0,1)),'constant')

    vec = p-xy
    vec = np.divide(vec,np.reshape(np.linalg.norm(vec,axis=1),(-1,1)))

    p =-2.7
    d = 0.22
    y=a*x**2+c  # function attempting to optimize
    z=math.sqrt(1-d**2)

    # ray method
    t=(-2*a*p*d+z-math.sqrt(z**2-4*a*p*d*z-4*a*d**2*c))/(2*a*d**2)  # distance from p to intersection point
    x_0 = p+t*d

    # geometric method
    m=z/d
    b=-m*p
    if d < 0: # {+-} based on if d and therefore m is positive/negative
      x_0 = (m + math.sqrt(m**2-4*a*(c-b)))/2*a  
    elif d == 0:
      x_0 = p
    else:
      x_0 = (m - math.sqrt(m**2-4*a*(c-b)))/2*a  

    n=1/(-2*a*x_0)  # slope of normal
    y=n*x-n*x_0+a*x_0**2+c  # eqation for line that is the normal

    r_slope = ((2*n) + (m*n**2) - m) / (2*n*m - n**2 + 1)  # from https://stackoverflow.com/questions/17395860/how-to-reflect-a-line-over-another-line
    y=r_slope*x-r_slope*x_0+a*x_0**2+c  # eqation for line that is the reflection



    p0 = 4
    p1 = 0
    p2 = -.057
    p4 = .0003
    p6 = -.00002
    p8 = .000003
    p10 = -.0000004
    p12 = .00000002
    x = np.roots([p12,0,p10,0,p8,0,p6,0,p4,0,p2,p1-m,p0-b])[-1]
    x = np.real(x) if np.imag(x) == 0 else None

    poly = np.poly1d([p12,0,p10,0,p8,0,p6,0,p4,0,p2,p1,p0])
    grad  = 1/-poly.deriv()(x)
    y = poly(x)

def mvpolyfit():
    # Given X, Y, and Z points, this will return coefficeints for a polynomial which fits
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y, copy=False)
    Z = X**2 + Y**2 + np.random.rand(*X.shape)*0.01

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B = Z.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)

def calcLineFromRay(rd, rp, axis, var):
    if axis == var:
        raise AttributeError('CalcLineFromRay: the axis cannot be the same as the variable')
    if rd[var] == 0:
        raise ValueError('CalcLineFromRay: Vector of variable cannot have 0 value')
    m = rd[axis]/rd[var]
    b = (rd[var]*rd[axis]*rp[axis]-rp[var]*rd[axis])/rd[var]
    return m, b

def reflectRayMVPolynomial(p, rd, rp):
    m,b = calcLineFromRay(rd,rp,1,0)
    lineY = np.poly1d([b,m])            # calc poly1ds for y of ray
    m,b = calcLineFromRay(rd,rp,2,0)
    lineZ = np.poly1d([b,m])            # calc poly1ds for y of ray
    x,y,z = intersectMVLinePolynomial(p, lineY, lineZ) # intersect line and poly
    n = getMVPolyNormal(p,x,y)          # find normal at intersection
    fd = rd - 2*n*(np.dot(rd,n))        # from http://paulbourke.net/geometry/reflected/
    fp = np.array([x,y,z])
    return fd, fp

def flattenMVPolynomial(p, val, axis):
    final = np.poly1d(())
    for m in range(p.shape[axis]):
        a = np.poly1d(np.take(p,m,axis))
        final += val**(d.shape[axis]-m-1)*a
    return final

def intersectMVLinePolynomial(p, lineY, lineZ):
    '''
      lines need to be poly1d
      poly needs to be poly2d
    '''
    # convert y's to x's
    # equate the z's (add the lineZ coeffs to poly1d)
    # find roots of poly1d
    # choose best real root as x
    # eval lineY(x) for y
    # eval lineZ(x) for z
    # return x,y,z

def getMVPolyNormal(p, x, y):
    '''
      calculate the normal vector for a 3d polynomial at x and y
    '''
    dx = polyder2d(p)                             # get derivative with respect to x
    gx = polyval2d(x,y,np.fliplr(np.flipud(dx)))  # evaluate for gradient wrt x - why, oh why, is polynomial package backwards from polyval and poly1d?
    gx = np.array((gx,0,1))                       # translate them to vectors
    dy = polyder2d(p,1)                           # get derivative with respect to y
    gy = polyval2d(x,y,np.fliplr(np.flipud(dy)))  # evaluate for gradient wrt y - why, oh why, is polynomial package backwards from polyval and poly1d?
    gy = np.array((0,gy,1))                       # translate them to vectors
    c = np.cross(gy,gx)                           # get cross product
    return c/norm(c)

def mvpolyval1d(p, val, axis=0):
    # ToDo: need to switch to Horner's method
    # why would we ever need to do this?
    truepoly = isinstance(p, poly2d)
    p = NX.asarray(p)
    o = p.shape[axis]
    o = np.reshape(np.power(val,NX.arange(o, 0, -1)-1),(-1,1))
    o = o if axis == 0 else o.T
    p = p * o
    y = np.sum(p,axis-1)
    if truepoly:
        y = poly1d(y)
    return y

def polyder2d(p, axis=0, m=1):
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")

    truepoly = isinstance(p, poly2d)
    p = NX.asarray(p)
    # calculate derivative
    n = p.shape[axis] - 1
    o = np.reshape(NX.arange(n, 0, -1),(-1,1))
    o = o if axis != 0 else o.T
    y = (np.take(p,np.arange(0,n),axis).T * o).T
    if m == 0:
        val = p
    elif m == 1:
        val = y
    else:
        val = polyder2d(y, axis, m - 1)
    if truepoly:
        val = poly2d(val)
    return val

class poly2d(object):
    """
      A two-dimensional polynomial class.

    """
    __hash__ = None

    @property
    def coeffs(self):
        """ A copy of the polynomial coefficients """
        return self._coeffs.copy()

    @property
    def variables(self):
        """ The name of the polynomial variable """
        return self._variables

    # calculated attributes
    @property
    def order(self):
        """ The order or degree of the polynomial """
        return max(len(self._coeffs) - 1,len(self._coeffs.T) - 1)

    '''  2d polynomials don't have roots, but maybe we could return the set of 1d polynomials where z = 0 ???
    @property
    def roots(self):
        """ The roots of the polynomial, where self(x) == 0 """
        return roots(self._coeffs)
    '''

    # our internal _coeffs property need to be backed by __dict__['coeffs'] for
    # scipy to work correctly.
    @property
    def _coeffs(self):
        return self.__dict__['coeffs']
    @_coeffs.setter
    def _coeffs(self, coeffs):
        self.__dict__['coeffs'] = coeffs

    # alias attributes
    c = coef = coefficients = coeffs
    o = order

    def __init__(self, c_or_p2d, variables=None):
        # ToDo: confirm variables is a 2 item tuple or list
        if isinstance(c_or_p2d, poly2d):
            self._variables = c_or_p2d._variables
            self._coeffs = c_or_p2d._coeffs

            if set(c_or_p2d.__dict__) - set(self.__dict__):
                msg = ("In the future extra properties will not be copied "
                       "across when constructing one poly1d from another")
                warnings.warn(msg, FutureWarning, stacklevel=2)
                self.__dict__.update(c_or_p2d.__dict__)

            if variables is not None:
                self._variables = variables
            return
        c_or_p2d = np.atleast_2d(c_or_p2d)
        if c_or_p2d.ndim > 2:
            raise ValueError("Polynomial must be 2d only.")
        #c_or_p2d = trim_zeros(c_or_p2d, trim='f')
        if len(c_or_p2d) == 0:
            c_or_p2d = NX.array([[0.]])
        self._coeffs = c_or_p2d
        if variables is None:
            variables = ('x','y')
        self._variables = variables

    def __array__(self, t=None):
        if t:
            return NX.asarray(self.coeffs, t)
        else:
            return NX.asarray(self.coeffs)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "poly2d(%s)" % vals

    def __len__(self):
        return self.order
'''
From here down needs to be converted to 2d
    def __str__(self):
        thestr = "0"
        var = self.variable

        # Remove leading zeros
        coeffs = self.coeffs[NX.logical_or.accumulate(self.coeffs != 0)]
        N = len(coeffs)-1

        def fmt_float(q):
            s = '%.4g' % q
            if s.endswith('.0000'):
                s = s[:-5]
            return s

        for k in range(len(coeffs)):
            if not iscomplex(coeffs[k]):
                coefstr = fmt_float(real(coeffs[k]))
            elif real(coeffs[k]) == 0:
                coefstr = '%sj' % fmt_float(imag(coeffs[k]))
            else:
                coefstr = '(%s + %sj)' % (fmt_float(real(coeffs[k])),
                                          fmt_float(imag(coeffs[k])))

            power = (N-k)
            if power == 0:
                if coefstr != '0':
                    newstr = '%s' % (coefstr,)
                else:
                    if k == 0:
                        newstr = '0'
                    else:
                        newstr = ''
            elif power == 1:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = var
                else:
                    newstr = '%s %s' % (coefstr, var)
            else:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = '%s**%d' % (var, power,)
                else:
                    newstr = '%s %s**%d' % (coefstr, var, power)

            if k > 0:
                if newstr != '':
                    if newstr.startswith('-'):
                        thestr = "%s - %s" % (thestr, newstr[1:])
                    else:
                        thestr = "%s + %s" % (thestr, newstr)
            else:
                thestr = newstr
        return _raise_power(thestr)
    
    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs * other)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __rmul__(self, other):
        if isscalar(other):
            return poly1d(other * self.coeffs)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __add__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __radd__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __pow__(self, val):
        if not isscalar(val) or int(val) != val or val < 0:
            raise ValueError("Power to non-negative integers only.")
        res = [1]
        for _ in range(val):
            res = polymul(self.coeffs, res)
        return poly1d(res)

    def __sub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(self.coeffs, other.coeffs))

    def __rsub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(other.coeffs, self.coeffs))

    def __div__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs/other)
        else:
            other = poly1d(other)
            return polydiv(self, other)

    __truediv__ = __div__

    def __rdiv__(self, other):
        if isscalar(other):
            return poly1d(other/self.coeffs)
        else:
            other = poly1d(other)
            return polydiv(other, self)

    __rtruediv__ = __rdiv__

    def __eq__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return (self.coeffs == other.coeffs).all()

    def __ne__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        return not self.__eq__(other)


    def __getitem__(self, val):
        ind = self.order - val
        if val > self.order:
            return 0
        if val < 0:
            return 0
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError("Does not support negative powers.")
        if key > self.order:
            zr = NX.zeros(key-self.order, self.coeffs.dtype)
            self._coeffs = NX.concatenate((zr, self.coeffs))
            ind = 0
        self._coeffs[ind] = val
        return

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1, k=0):
        """
        Return an antiderivative (indefinite integral) of this polynomial.
        Refer to `polyint` for full documentation.
        See Also
        --------
        polyint : equivalent function
        """
        return poly1d(polyint(self.coeffs, m=m, k=k))

    def deriv(self, axis, m=1):
        """
        Return a derivative of this polynomial.
        Refer to `polyder` for full documentation.
        See Also
        --------
        polyder : equivalent function
        """
        return poly1d(polyder2d(self.coeffs, axis, m=m))
'''
