this is the derivation of the low-rank optimization problem, and some notes for 'unfolding to network'

### derivation	

the optimization problem is:
$$
\min_X \frac 12 \Vert SFX-b\Vert_F^2+\lambda \Vert X \Vert_*
$$
where $X$ is the Cine MR image, $b$ is the under-sampled k-space data, $F$ is the Fourier transform in the spatial axis, $S$ is the under-sampling mask.

it can be solved based on the following variable splitting algorithm：
$$
\begin{align}
& \min_X \frac 12 \Vert SFX-b\Vert_F^2+\lambda \Vert A \Vert_* \\
& s.t. \quad A = X
\end{align}
$$
using ADMM, result in iterative scheme:
$$
\left\{
\begin{align}
& A,X =  \arg \min_{A,X} \frac 12 \Vert SFX-b\Vert_2^2+\lambda \Vert A \Vert_* + \frac {\mu}2\Vert A-X-L\Vert_2^2 \\
& L = L-(A-X)
\end{align}
\right.
$$
then,
$$
\left\{
\begin{align}
&A =  \arg\min_A \lambda \Vert A \Vert_* + \frac {\mu}2\Vert A-X-L\Vert_2^2 \\
&X =  \arg \min_{X} \frac 12 \Vert SFX-b\Vert_2^2 + \frac {\mu}2\Vert A-X-L\Vert_2^2 \\
& L = L-(A-X)
\end{align}
\right.
$$
the $A$ subproblem can be solved by singular value thresholding method, and the $X$ subproblem is a quadratic problem, which can be solved by taking derivation of $X$ and let it equal to zero. thus,
$$
\left\{
\begin{align}
&A =  SVT(X+L,\frac {\lambda}{\mu}) \\
&X =  F*\frac{S^*b+\mu F(A-L)}{S^*S+\mu} \\
& L = L-(A-X)
\end{align}
\right.
$$

### NOTE

to be continued.....