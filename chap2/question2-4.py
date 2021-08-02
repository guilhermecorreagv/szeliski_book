import matplotlib.pyplot as plt
import numpy as np
'''
Ex 2.4: Focus distance and depth of field Figure out how the focus distance and depth of
field indicators on a lens are determined.

1. Compute and plot the focus distance z_o as a function of the distance traveled from the
focal length ∆z_i = f − z_i for a lens of focal length f (say, 100mm). Does this explain
the hyperbolic progression of focus distances you see on a typical lens (Figure 2.20)?

3. Now consider a zoom lens with a varying focal length f . Assume that as you zoom,
the lens stays in focus, i.e., the distance from the rear nodal point to the sensor plane
z_i adjusts itself automatically for a fixed focus distance z_o . How do the depth of field
indicators vary as a function of focal length? Can you reproduce a two-dimensional
plot that mimics the curved depth of field lines seen on the lens in Figure 2.20b?
'''


def exercise1():
    # So we have the lens equation 1/z_o + 1/z_i = 1/f
    fs = [25, 50, 100, 200, 400]
    for f in fs:
        dz = np.linspace(-99, 99, 1000, dtype='float')
        zo = 1 / (1 / f - 1 / (f - dz))
        plt.plot(dz, zo)
    legends = ['f=' + str(f) for f in fs]
    plt.ylim([-5000, 5000])
    plt.xlabel(r'$\Delta z_i$')
    plt.ylabel(r'$z_o$')
    plt.legend(legends)
    plt.show()


def exercise3():
    '''
        DOF =  \frac{2Nf^2z_oc(z_o-f)}{f^4 + c^2N^2(z_o-f)^2}
        Note that we cannot have a f that's bigger than z_o
    '''
    zo = 2000  # 2 meters
    N = 8
    f = np.linspace(10, 1600, 1000)
    c = 20
    DOF = (2 * N * f * f * zo * c * (zo - f)) / (f**4 + c * c * N * N *
                                                 (zo - f) * (zo - f))

    plt.plot(f, DOF)
    plt.title(r'Curve for $z_o=2000$, $N=8$, $c=20$')
    plt.xlabel('Focus Distance (mm)')
    plt.ylabel('DOF (mm)')
    plt.show()


if __name__ == '__main__':
    exercise3()