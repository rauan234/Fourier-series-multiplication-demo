import matplotlib.pyplot as plt
from numpy import cos


neglect_high_harmonics = False


def plot(f, start, end, N, **kwargs):
    x = start

    step = (end - start) / N

    xlist = []
    ylist = []
    for iteration in range(N):
        xlist.append(x)
        ylist.append(f(x))
        
        x += step

    plt.plot(xlist, ylist, **kwargs)

def Sum(f, low, high, *args):
    out = 0

    for x in range(low, high + 1):
        out += f(x, *args)
        
    return out


class Fourier:
    def __init__(self, coefs, omega):
        if not(isinstance(omega, int) or isinstance(omega, float)):
            raise TypeError

        if not isinstance(coefs, list):
            raise TypeError

        for coef in coefs:
            if not(isinstance(coef, int) or isinstance(coef, float)):
                raise TypeError
            
        
        self.coefs = coefs
        self.omega = omega

    def calc(self, x):
        return Sum(lambda n, C, omega: C[n] * cos(n * omega * x),
                   0, len(self.coefs) - 1, self.coefs, self.omega)

    def mul_err(u, v):
        return Sum(lambda g, a, b: Sum(lambda h, a, b: a[g + h + 1] * b[len(a) - 1 - h],
                                       0, len(a) - 2 - g, a, b),
                   0, len(u.coefs) - 2, u.coefs, v.coefs) / 2

    def __mul__(this, other):
        if(this.omega != other.omega):
            raise ValueError

        if(len(this.coefs) != len(other.coefs)):
            raise ValueError

        res_coefs = []

        res_coefs.append(Sum(lambda n, a, b: a[n]  * b[n], 1, len(this.coefs) - 1, this.coefs, other.coefs) / 2 +
                         this.coefs[0] * other.coefs[0])
        
        for k in range(1, len(this.coefs)):
            res_coefs.append(
                Sum(lambda n, a, b: (a[n] * b[n + k] + a[n + k] * b[n]), 0, len(this.coefs) - 1 - k, this.coefs, other.coefs) / 2 +
                Sum(lambda n, a, b: a[n] * b[k - n], 0, k, this.coefs, other.coefs) / 2
                )

        if not neglect_high_harmonics:
            for k in range(len(this.coefs), 2 * len(this.coefs) - 1):
                res_coefs.append(
                    Sum(lambda n, a, b: a[n] * b[k - n] / 2, k - len(this.coefs) + 1, len(this.coefs) - 1, this.coefs, other.coefs)
                    )

        return Fourier(res_coefs, this.omega)


u = Fourier([1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/16], 1)
v = Fourier([9, -8, 7, 6, 5, -4, 3], 1)

f = u * v

if neglect_high_harmonics:
    print('Maximal error due to multiplication:', Fourier.mul_err(u, v))


plot(lambda x: u.calc(x) * v.calc(x), 0, 4, 1000, color='g')
plot(lambda x: f.calc(x), 0, 4, 1000, color='#000000', linestyle='dashed')

if neglect_high_harmonics:
    plt.title('f(x) ~ u(x) * v(x);      ' +
              'error: ' + str(round(Fourier.mul_err(u, v), 2)))

else:
    plt.title('f(x) = u(x) * v(x)')
    
plt.legend(['u(x) * v(x)', 'f(x)'])
plt.xlabel('x')
plt.ylabel('y')

plt.show()
