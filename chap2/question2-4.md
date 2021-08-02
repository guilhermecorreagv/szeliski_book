# Question 2.4: Focus distance and depth of field Figure out how the focus distance and depth of field indicators on a lens are determined. 

## 2. Compute the depth of field (minimum and maximum focus distances) for a given focus setting z_o as a function of the circle of confusion diameter c (make it a fraction of the sensor width), the focal length f , and the f-stop number N (which relates to the aperture diameter d). Does this explain the usual depth of field markings on a lens that bracket the in-focus marker, as in Figure 2.20a?

We have initially a focused image with distance from the lens to image plane given by $z_i$, then we change the focus so we have a circle of confusion $c$. The new $z_i'$ is given by $z_i + \Delta z_i$. 

We have then:

$$
    z_i = \frac{f}{1 - f/z_o} \\ 
    z_i' = \frac{f}{1 - f/z_o'}
$$

First, from the biggest distance we have the following triangle similarity:

$$
    \frac{c}{z_i - z_i'} = \frac{d}{z_i'} \\
    c = d \left( \frac{z_i}{z_i'} - 1\right) \\
    z_o' = \frac{dfz_o}{df + c(f - z_o)}
$$

By similar arguments we find that the minimum $z_o$ is:

$$
    z_o'' = \frac{dfz_o}{df + c(z_o - f)}
$$

The depth of field is the difference between them $DOF = z_o' - z_o''$, then we have:

$$
    DOF = \frac{dfz_o}{df + c(f - z_o)} - \frac{dfz_o}{df + c(z_o - f)} \\
    DOF = \frac{2dfz_oc(z_o-f)}{d^2 f^2 + c^2 (z_o-f)^2} \text{, using that $d=f/N$ we have} \\
    DOF =  \frac{2Nf^2z_oc(z_o-f)}{f^4 + c^2N^2(z_o-f)^2}
$$

<!-- 


$$
    \frac{d}{z_i} = \frac{c}{z_i - \Delta z_i} \\ 
    c = d \left( 1 - \frac{\Delta z_i}{z_i} \right) 
$$ 

Now using the lens equation we also have:
$$
    z_i = \frac{z_o f}{z_o - f}
$$

By using the f-stop number we have $d=f/N$, and we also use c as a fraction of the width, so $c = \alpha W$, with $\alpha \in [0,1]$.

$$
    \alpha W = \frac{f}{N} \left( 1 - \frac{\Delta z_i}{z_i} \right) 
$$


Now we substitute for $\Delta z_i$:

$$
    \Delta z_i = \vert z_i - f \vert \\
    = \frac{z_o f}{z_o - f} - f \\
    = \frac{f^2}{z_o - f}
$$

Back on the original equation:

$$
    c = d \left( 1 - \frac{f}{z_o} \right)
$$

By using the f-stop number we have $d=f/N$, and we also use c as a fraction of the width, so $c = \alpha W$, with $\alpha \in [0,1]$.

$$
    \alpha W = \frac{f}{N} \left( 1 - \frac{f}{z_o} \right)
$$

We want to have a confusion diameter of less than $\alpha W$, then we have:

$$
    \frac{f}{N} \left( 1 - \frac{f}{z_o} \right) < \alpha W \\

$$ -->