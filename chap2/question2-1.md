## Question

**Ex 2.1: Least squares intersection point and line fitting—advanced Equation (2.4) shows how the intersection of two 2D lines can be expressed as their cross product, assuming the lines are expressed as homogeneous coordinates.**

1.  If you are given more than two lines and want to find a point $\tilde{x}$ that minimizes the sum of squared distances to each line,
    $$
        D = \sum_i \left( \boldsymbol{\tilde{x}} \cdot \boldsymbol{\tilde{l}}_i \right)^2
    $$
    how can you compute this quantity? (Hint: Write the dot product as $\boldsymbol{\tilde{x}}^T \boldsymbol{\tilde{l}}_i$ and turn the squared quantity into a quadratic form, $\boldsymbol{\tilde{x}}^T A \boldsymbol{\tilde{x}}$.)

2.  To fit a line to a bunch of points, you can compute the centroid (mean) of the points as well as the covariance matrix of the points around this mean. Show that the line passing through the centroid along the major axis of the covariance ellipsoid (largest eigenvector) minimizes the sum of squared distances to the points.

3.  These two approaches are fundamentally different, even though projective duality tells us that points and lines are interchangeable. Why are these two algorithms so apparently different? Are they actually minimizing different objectives?


## Solution

1.  Following the hint given by the author we have:
    $$
        D = \sum_i \boldsymbol{\tilde{x}}^T \boldsymbol{\tilde{l}}_i \boldsymbol{\tilde{x}}^T \boldsymbol{\tilde{l}}_i \\
        D = \sum_i \boldsymbol{\tilde{x}}^T \boldsymbol{\tilde{l}}_i \left(  \boldsymbol{\tilde{l}}_i^T \boldsymbol{\tilde{x}} \right)^T \\ 
        \text{since this is a dot product, the term inside the parenthesis is equal to it's transpose} \\
        D = \sum_i \boldsymbol{\tilde{x}}^T \boldsymbol{\tilde{l}}_i \boldsymbol{\tilde{l}}_i^T \boldsymbol{\tilde{x}} \\
        D = \sum_i \boldsymbol{\tilde{x}}^T A_i \boldsymbol{\tilde{x}} \\
        D = \boldsymbol{\tilde{x}}^T \left( \sum_i A_i \right) \boldsymbol{\tilde{x}} \\
        D = \boldsymbol{\tilde{x}}^T A \boldsymbol{\tilde{x}}
    $$

    Where
    $$
        A = \sum_i \boldsymbol{\tilde{l}}_i \boldsymbol{\tilde{l}}_i^T
    $$

    To avoid the trivial solution we'll restrict ourselves to the case where $\vert \vert x \vert \vert_2 = 1$. We also note that by our construction A is semi-positive definite and symmetric. We then build the following Lagrangian:

    $$
        \mathcal{L}(\boldsymbol{\tilde{x}}, \lambda) = \boldsymbol{\tilde{x}}^T A \boldsymbol{\tilde{x}} + \lambda(\boldsymbol{1 - \tilde{x}}^T \boldsymbol{\tilde{x}})
    $$

    We have then:
    $$
    \frac{\partial  \mathcal{L}(\boldsymbol{\tilde{x}}, \lambda)}{\partial \boldsymbol{\tilde{x}}} = 2 A \boldsymbol{\tilde{x}} - 2 \lambda \boldsymbol{\tilde{x}} \\
    ( A - \lambda I ) \boldsymbol{\tilde{x}} = 0 
    $$

    We notice that $\boldsymbol{\tilde{x}}$ is an eigenvector of A, and thus we have:
    $$
        \boldsymbol{\tilde{x}}^T A \boldsymbol{\tilde{x}} = \boldsymbol{\tilde{x}}^T \lambda^* \boldsymbol{\tilde{x}} \\
        = \lambda^*
    $$

    So the solution is the eigenvector associated with the smallest eigenvalue of A. 

2.  We have a matrix $\boldsymbol{X}$ where each row is a point. Then the centroid $\boldsymbol{\bar{x}}$ is:
    $$
        \boldsymbol{\bar{x}} = \frac{1}{n} \sum_i \boldsymbol{X}_i
    $$

    We then extend this average row by doing:
    $$
        \boldsymbol{\bar{X}} = \boldsymbol{1}_{n \times 1} \boldsymbol{\bar{x}}
    $$

    Now we have our zero-mean points:
    $$
        B = \boldsymbol{X} - \boldsymbol{\bar{X}}
    $$

    We want to find the eigenvector associated with the highest eigenvalue of the covariance matrix, then we want:
    $$
        v_1 = \underset{\vert \vert v \vert \vert_2 = 1}{\operatorname{argmax}} (v^T B^T B v)
    $$

    Now, we exploit the fact that by translating the points, the distance between them remain unchanged. With this in mind, we can focus our attention to the matrix B, and consider the lines that have the component $c=0$, since we want them to pass through the centroid (which now is the origin). Our new condition becomes:

    $$
        v^* = \underset{\vert \vert v \vert \vert_2 = 1}{\operatorname{argmin}} (\sum_i dist(B_i, proj_{v}B_i)^2) \\
        v^* = \underset{\vert \vert v \vert \vert_2 = 1}{\operatorname{argmin}} (\sum_i B_iB_i^T - (B_i \cdot v)^2) \\
        v^* = \underset{\vert \vert v \vert \vert_2 = 1}{\operatorname{argmax}} (\sum_i (B_i \cdot v)^2)\text{, since v only appears in the second term} \\
        v^* = \underset{\vert \vert v \vert \vert_2 = 1}{\operatorname{argmax}} (\sum_i (B_i v)^2) \text{Recall that $B_i$ is a line vector, and $v$ is a column vector} \\
        v^* = \underset{\vert \vert v \vert \vert_2 = 1}{\operatorname{argmax}} (\sum_i v^T B_i^T B_i v) \\
        v^* = \underset{\vert \vert v \vert \vert_2 = 1}{\operatorname{argmax}} (v^T (\sum_i B_i^T B_i) v) \\
        v^* = \underset{\vert \vert v \vert \vert_2 = 1}{\operatorname{argmax}} (v^T B^T B v)
    $$

    Which is the definition of $v_1$.