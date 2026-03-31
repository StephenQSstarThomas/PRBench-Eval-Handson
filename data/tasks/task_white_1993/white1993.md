# Density-matrix algorithms for quantum renormalization groups

*Converted from: Density-matrix algorithms for quantum renormalization groups.pdf*

---

<!-- Page 1 -->

PHYSICAL REVIEW B
VOLUME 48, NUMBER 14
1 OCTOBER 1993-II

# Density-matrix algorithms for quantum renormalization groups

Steven R. White
*Department of Physics, University of California, Irvine, California 92717*
(Received 13 January 1993)

A formulation of numerical real space renormalization groups for quantum many-body problems is presented and several algorithms utilizing this formulation are outlined. The methods are presented and demonstrated using $S = 1/2$ and $S = 1$ Heisenberg chains as test cases. The key idea of the formulation is that rather than keep the lowest-lying eigenstates of the Hamiltonian in forming a new effective Hamiltonian of a block of sites, one should keep the most significant eigenstates of the block density matrix, obtained from diagonalizing the Hamiltonian of a larger section of the lattice which includes the block. This approach is much more accurate than the standard approach; for example, energies for the $S = 1$ Heisenberg chain can be obtained to an accuracy of at least $10^{-9}$. The method can be applied to almost any one-dimensional quantum lattice system, and can provide a wide variety of static properties.

## I. INTRODUCTION

Shortly after Wilson developed his numerical renormalization group (RG) procedure to solve the Kondo problem[1], there was considerable interest in applying closely related techniques to a variety of problems. In particular, it seemed that a number of quantum lattice models (such as the Hubbard and Heisenberg models), particularly in one dimension (1D), could be treated with a real-space blocking version of this technique. It was clear from the beginning that one could not hope to achieve the accuracy Wilson obtained for the Kondo problem in these other systems (cf. Sec. III), but it was hoped that the method would yield qualitatively reliable results. Unfortunately, the approach proved to be rather unreliable, particularly in comparison with other numerical approaches, such as Monte Carlo, which were being developed at the same time. Until very recently, the method was only used occasionally.

Recent developments in renormalization group algorithms have changed this picture completely. The first significant advance came in the understanding of the effect of boundary conditions on the basic renormalization group procedure[2]. The standard approach of neglecting all connections to neighboring blocks during the diagonalization of the block Hamiltonian introduces large errors which cannot be corrected by any reasonable increase in the number of states kept. However, by varying the boundary conditions on a block, and keeping states from several diagonalizations with different boundary conditions, one can eliminate these errors, at least for single-particle problems. The second advance came in the development of a technique suitable for many-particle systems, using a formulation in terms of density matrices[3]. For systems such as 1D Heisenberg spin chains, the density-matrix approach makes the numerical renormalization group approach not just qualitatively reliable, it makes it substantially more accurate and powerful for calculating many zero temperature properties than current quantum Monte Carlo approaches.

The main goal of this paper is to describe in some detail the density-matrix algorithms. The approach was presented in summary form in an earlier paper[3], but in insufficient detail to allow a nonexpert reader to develop his own programs for these calculations. These algorithms are somewhat more complicated than typical exact diagonalization or Monte Carlo algorithms, and a number of unfamiliar ideas are involved. By reading this paper, we hope that someone with some experience in exact diagonalization calculations can develop their own density-matrix program. We will illustrate and demonstrate the approach on $S = 1/2$ and $S = 1$ Heisenberg antiferromagnetic spin chains. We will present a few new results for properties of these chains, but only as examples of the power of the methods. More results on the properties of Heisenberg chains will appear in subsequent papers.

In the next section, we discuss the standard real-space renormalization group approach. In Sec. III, we discuss previous uses of the standard methods. In Sec. IV, we then present the density-matrix approach, and show that it is optimal in a certain sense. In Sec. V, we present the infinite and finite lattice density-matrix algorithms. In Sec. VI, we discuss the measurement of a number of static quantities in these algorithms. In Sec. VII, we present some results for Heisenberg chains and discuss the effects of various boundary conditions on the procedure. In Sec. VIII, we conclude and discuss future prospects.

## II. STANDARD APPROACH

We first describe in detail the standard RG approach in the simplest possible context, a real-space blocking approach for a 1D lattice system. The notation and many of the central ideas will be very similar in the density-matrix approach described later. The Hamiltonian considered here could describe a spin system, such as the Heisenberg model, or an interacting electron system, such as the Hubbard model. The approach is relevant for zero temperature, and one obtains the ground state and some

0163-1829/93/48(14)/10345(12)/$06.00
48
10 345
©1993 The American Physical Society

---

<!-- Page 2 -->

10 346
STEVEN R. WHITE
48

low-lying excited states.

One begins by breaking the 1D chain into finite identical blocks. It is usually convenient to start at the first iteration with blocks consisting of just one site. We will label the blocks $B$ and the block Hamiltonian $H_B$. $H_B$ contains all terms of $H$ involving only sites contained in $B$. For example, for the Hubbard model at the first iteration, where $B$ consists of one site, $H_B = U n_{i\uparrow} n_{i\downarrow} - \mu (n_{i\uparrow} + n_{i\downarrow})$. For the Heisenberg model at the first iteration, $H_B = 0$.

Rather than describe $B$ and $H_B$ in the usual way by listing the sites of $B$ and using second-quantized operator expressions for $H_B$, we describe $B$ by a list of the many-body states on the block, and by quantum numbers and matrix elements between these states. We store the number of states $m$, and for each state we list all quantum numbers which are to be used, such as $S_z$ and $S$ for a spin system, or $N_\uparrow, N_\downarrow$, and $S$ for an electron system. $H_B$ is represented as an $m \times m$ matrix. In order to reconstruct $H$, additional information is needed besides $H_B$. The additional information describes the interactions between blocks. For a Heisenberg system with interaction
$$
\mathbf{S}_i \cdot \mathbf{S}_{i+1} = S_i^z S_{i+1}^z + \frac{1}{2}(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+), \tag{1}
$$
one needs to store $m \times m$ matrix representations of $S_i^z$, $S_i^+$, and $S_i^-$ for $i$ equal to both the left and right end sites of $B$. (In practice, one need not store $S_i^-$, since it can be obtained by taking the Hermitian conjugate of $S_i^+$.) For a Hubbard model one would have to store matrices for $c_{i\sigma}^\dagger$ and $c_{i\sigma}$, with $\sigma = \uparrow$ and $\downarrow$, in order to reconstruct the hopping term $\sum_\sigma (c_{i+1\sigma}^\dagger c_{i\sigma} + c_{i\sigma}^\dagger c_{i+1\sigma})$.

The standard procedure is summarized in Table I. At the beginning of an iteration one forms the Hamiltonian for two blocks joined together, $H_{BB}$. $BB$ has $m^2$ states. The states are labeled by 2 indices, $i_1 i_2$. For a Heisenberg system with $J = 1$ the $m^2 \times m^2$ matrix for $H_{BB}$ is given by
$$
\begin{aligned}
[H_{BB}]_{i_1 i_2; i_1' i_2'} &= [H_B]_{i_1 i_1'} \delta_{i_2 i_2'} + [H_B]_{i_2 i_2'} \delta_{i_1 i_1'} \\
&+ [S_r^z]_{i_1 i_1'} [S_l^z]_{i_2 i_2'} + \frac{1}{2}[S_r^+]_{i_1 i_1'} [S_l^-]_{i_2 i_2'} \\
&+ \frac{1}{2}[S_r^-]_{i_1 i_1'} [S_l^+]_{i_2 i_2'}
\end{aligned} \tag{2}
$$
where $r$ represents the rightmost site of the left block, and $l$ the leftmost site of the right block.

In diagonalizing $H_{BB}$ it is useful to separate the basis states by quantum numbers, since $H_{BB}$ is block diagonal. It is very simple to use $S_z$ or $N_\uparrow$ and $N_\downarrow$ in this way. Utilizing the total spin $S$ is more tedious (especially when one puts four blocks together, as we do below), and we have not used $S$ to further reduce the dimension of $H_{BB}$. (The value of $S$ for a state can easily be inferred by degeneracies for different values of $S_z$.)

The lowest-lying eigenstates $u_i^\alpha$, $\alpha = 1, \ldots, m$, of $H_{BB}$ are the states used to describe $B'$ ($BB \to B'$). The new block Hamiltonian matrix $H_{B'}$ is diagonal. However, in the more general case where the states kept, the $u^\alpha$, are not eigenstates of $H_{BB}$ we can write
$$
H_{B'} = O H_{BB} O^\dagger, \tag{3}
$$
where the $m \times m^2$ matrix $O_{i; i_1 i_2} = u_{i_1 i_2}^i$; i.e., the rows of $O$ are the states kept. If $O$ were square, this would be a unitary transformation. Since $O$ is not square, the transformation truncates away (integrates out) the high energy states.

In order to obtain new matrices for $S_\ell^z$, $S_r^z$, etc., it is necessary to use $O$ again. First, one must construct the operators for $S_\ell^z$, $S_r^z$, etc., for $BB$, which we denote by $\tilde{S}_\ell^z$, $\tilde{S}_r^z$, etc. For example
$$
[\tilde{S}_\ell^z]_{i_1 i_2; i_1' i_2'} = [S_\ell^z]_{i_1 i_1'} \delta_{i_2 i_2'}, \tag{4}
$$
$$
[\tilde{S}_r^z]_{i_1 i_2; i_1' i_2'} = [S_r^z]_{i_2 i_2'} \delta_{i_1 i_1'}. \tag{5}
$$
Then the new matrices for $B'$ are given by
$$
S_\ell^z = O \tilde{S}_\ell^z O^\dagger, \tag{6}
$$
etc.

After these new operator matrices are formed, we can replace $B$ by $B'$ and start the next iteration. The iteration is continued until the system is large enough to represent properties of the infinite system. As our main concern here is the iterative diagonalization procedure discussed above, we will not discuss the analysis, using fixed points, relevant and irrelevant operators, etc., of the effective Hamiltonians obtained with the procedure.

### III. PREVIOUS USES

We will not attempt to give a comprehensive review of previous uses of this approach; rather, we will discuss a handful of studies which illustrate successful and unsuccessful applications.

Wilson's approach to the Kondo problem is closely related to the method described here, despite some important differences. One difference is that rather than joining two identical blocks, the degrees of freedom as-

TABLE I. Standard numerical renormalization group algorithm for a 1D quantum system.

| Step | Description                                                                                                                                                                              |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.   | Isolate two blocks $BB$, and form $H_{BB}$.                                                                                                                                              |
| 2.   | Diagonalize $H_{BB}$, obtaining the $m$ lowest eigenvectors $u^\alpha$.                                                                                                                  |
| 3.   | Form matrix representations of $S_\ell^z$, etc., for $BB$ from the corresponding matrices for $B$.                                                                                       |
| 4.   | Change basis to the $u^\alpha$, keeping only the lowest $m$ states, using $H_{B'} = O H_{BB} O^\dagger$, etc., with $O(\alpha; i_1, i_2) = u_{i_1 i_2}^\alpha$, $\alpha = 1, \ldots, m$. |
| 5.   | Replace $B$ with $B'$.                                                                                                                                                                   |
| 6.   | Go to step 1.                                                                                                                                                                            |

---

<!-- Page 3 -->

48
DENSITY-MATRIX ALGORITHMS FOR QUANTUM . . .
10 347

sociated with a single interval (an "onion layer") were added to the system at each iteration. The analogous procedure for a 1D system would be to add a single site to a block at each iteration. From a computational point of view, this has a distinct advantage in that many more states can be kept ($m$ can be made larger) since at each iteration a system with $nm$ states, as opposed to $m^2$, must be diagonalized, where $n$ is the number of states on a single site ($n = 4$ for Hubbard models, $n = 2S + 1$ for spin models).

The most important difference between the Kondo system and a 1D system is that the couplings between adjacent layers or "sites" decreases exponentially in the Kondo system, whereas it remains constant for a 1D system. This exponential decrease is the key to the success of the method for the Kondo system and related impurity systems. More discussion concerning how the detailed form of the Hamiltonian makes the numerical approach accurate are given by Wilson.[1]

Bray and Chui[4] applied the approach described in the previous section to the 1D Hubbard model. The results were quite discouraging. Even when a large number of states were kept ($m \approx 1000$), results for the energies of the lowest few sites were off by 5--10 % for 16 site chains. Results for 32 sites or larger were not considered reliable and were not presented. The origin of the difficulties was not clearly understood.

Other uses of this general approach since then have included the work of Pan and Chen$^5$ and Kovarik.[6] Very recently Xiang and Gehring[7] applied a slight variation of the method of the previous section to the 1D Heisenberg model. Their method added a single site to a block at each iteration, rather than doubling the block size each time. (The algorithms we present below also add a single site to a block.) This improvement gave results rather more encouraging than those of Bray and Chui; keeping about 200 states they obtained an error of about 0.5% in the ground state energy. This can be compared to our results below, where keeping fewer states we obtain the ground state energy to at least nine digits.

There have also been a number of analytical real-space RGs applied to the 1D Hubbard and related models.[8] These methods keep only a handful of states, and one generally only expects to obtain qualitative features. These will not be discussed further here.

Lee used a numerical RG method to study 2D Anderson localization.[9] Since the model studied was noninteracting, a two-dimensional calculation was feasible. It was clear from an examination of the density of states that the algorithm was not especially accurate. However, Lee argued that the errors would simply result in the system being in another realization of the random potential. Lee concluded that there was a critical amplitude for the random potential which induced localization. However, not long after Lee's work, Lee and Fisher found[10] that the 2D model is logarithmically localized even for arbitrarily small randomness using a different approach. This result is now generally accepted.

In summary, the standard numerical RG approach generally performs poorly. The exceptions to this rule involve Hamiltonians which can be put into a special form.

## IV. DENSITY-MATRIX APPROACH

The fundamental difficulty in the standard approach discussed in Sec. II lies in choosing the eigenstates of $H_{BB}$ to be the states kept. Since $H_{BB}$ contains no connections to the rest of the lattice, its eigenstates have inappropriate features at the block ends. This is clearly illustrated in the work of White and Noack,[2] who suggested two alternatives to the standard approach. These methods shared a common feature: The states that were kept were not the eigenstates of $H_{BB}$. They differed in how the states to be kept were chosen. In the first method, the combination of boundary conditions (CBC) approach, the lowest-lying eigenstates of several different block Hamiltonians were kept. The several block Hamiltonian differed only in the boundary condition applied to a block; e.g., one Hamiltonian might have periodic boundary conditions applied and another antiperiodic. The rationale for this was that quantum fluctuations in the rest of the system effectively apply a variety of boundary conditions to the block. States from any single boundary condition cannot respond properly to these fluctuations. By applying a representative set of boundary conditions, which is in some sense "complete" enough for the problem at hand, one obtains a set of states which are able to respond to these fluctuations. This approach proved very effective for the simple single-particle problems studied by White and Noack, as well as for Anderson localization models.[11]

The CBC approach appears to be ill suited to interacting systems. It is useful to consider a noninteracting many-particle system, such as the Hubbard model with $U = 0$. An arbitrary state of this system can be described in terms of the single-particle wave functions of each of its particles. Some of these single-particle wave functions may have nodes at the ends of a block, and some may have antinodes.

It is easy to choose boundary conditions with generate block states where every particle on the block has a node or every particle has an antinode, but it is difficult to get different boundary behavior for different particles. In order to properly represent the block, the states kept not only need to allow for different end behavior for different particles; they must represent a complete range of boundary behavior.

This general line of reasoning is supported by numerical tests on Heisenberg chains. We have tried to find a simple set of boundary conditions which can be used to treat the $S = \frac{1}{2}$ Heisenberg chain. We tried combinations of periodic and antiperiodic couplings between the ends of the block, as well as varying the magnitude of the coupling between the ends of a block. We were unable to find any set of boundary conditions which was at all satisfactory.

The other approach suggested by White and Noack, the superblock method, forms the basis for the density matrix approach. In the superblock method, one diagonalizes a larger system (the "superblock"; the name is analogous to "supercell," as used in electronic structure calculations) composed of three or more blocks which includes the two blocks $BB$ which are used to form $B'$. The

---

<!-- Page 4 -->

10 348
STEVEN R. WHITE
48

wave functions for the superblock are projected onto $BB$, and these projected states of $BB$ are kept. For a single-particle wave function, this projection is single valued and trivial. The superblock method works quite well in the single-particle model, with the accuracy increasing rapidly with the number of extra blocks used. However, for a many-particle wave function, the "projection" of a wave function onto $BB$ is many valued, and, in fact, a single many-particle state for the entire lattice generally "projects" onto a complete set of block states. However, some of these states are more important than others; the density matrix tells us which states are the most important. (The reader is urged to review Feynman's introduction to density matrices[12] before proceeding further.)

It is very natural to use the density matrix to choose the states which we wish to keep. Consider first the following argument by analogy. For an isolated block at finite temperature, the probability that the block is in an eigenstate $\alpha$ of the block Hamiltonian is proportional to its Boltzmann weight $\exp(-\beta E_\alpha)$. The Boltzmann weight is an eigenvalue of the density matrix $\exp(-\beta H_B)$, and an eigenstate of the Hamiltonian is also an eigenstate of the density matrix. Since lowest energy corresponds to highest probability in the Boltzmann weight, we can view the standard RG approach as choosing the $m$ most probable eigenstates to represent the block given the assumption that the block is isolated. (Alternatively, we can view the rest of the lattice as a heat bath at an effective inverse temperature $\beta$, to which the system is very weakly coupled.) However, in reality the block is not isolated, the density matrix is not $\exp(-\beta H_B)$ [it is defined through Eq. (12) or (16) below], and eigenstates of the block Hamiltonian are not eigenstates of the block's density matrix. For a system which is strongly coupled to the outside universe, it is much more appropriate to use the eigenstates of the density matrix to describe the system rather than the eigenstates of the system's Hamiltonian. Thus a natural generalization of the standard approach is to choose to keep the $m$ most probable eigenstates of the block density matrix.

This argument can be made much more precise. In particular, we can show that keeping the most probable eigenstates of the density matrix gives the most accurate representation of the state of the system as a whole, i.e., the block plus the rest of the lattice. Let us assume we have diagonalized a superblock and obtained one particular state $|\psi\rangle$, probably the ground state. Let $|i\rangle$, $i = 1,\ldots,\ell$ be a complete set of states of $BB$ (the system) and $|j\rangle$, $j = 1,\ldots,J$ be the states of the rest of the superblock, i.e., the "universe." We can write $|\psi\rangle = \sum_{i,j} \psi_{ij}|i\rangle|j\rangle$. We will assume for simplicity $\psi_{ij}$ is real. We wish to define a procedure for producing a set of states of the system $|u^\alpha\rangle$, $\alpha = 1,\ldots,m$, with $|u^\alpha\rangle = \sum_i u_i^\alpha|i\rangle$, which are optimal for representing $\psi$ in some sense. Because we allow only $m$ states, we cannot represent $|\psi\rangle$ exactly if $\ell > m$. We wish to construct an accurate expansion for $|\psi\rangle$ of the form
$$
|\psi\rangle \approx |\bar{\psi}\rangle = \sum_{\alpha,j} a_{\alpha,j}|u^\alpha\rangle|j\rangle. \tag{7}
$$
In other words, we wish to minimize
$$
\mathcal{S} = \left||\psi\rangle - |\bar{\psi}\rangle\right|^2, \tag{8}
$$
by varying over all $a_{\alpha,j}$ and $u^\alpha$, subject to $\langle u^\alpha|u^{\alpha'}\rangle = \delta_{\alpha\alpha'}$. Without loss of generality, we can write
$$
|\bar{\psi}\rangle = \sum_\alpha a_\alpha |u^\alpha\rangle|v^\alpha\rangle, \tag{9}
$$
where $v_j^\alpha = \langle j|v^\alpha\rangle = N_\alpha a_{\alpha,j}$, with $N_\alpha$ chosen to set $\sum_j |v_j^\alpha|^2 = 1$. Switching to matrix notation, we have
$$
\mathcal{S} = \sum_{ij} \left(\psi_{ij} - \sum_{\alpha=1}^m a_\alpha u_i^\alpha v_j^\alpha\right)^2, \tag{10}
$$
and we minimize $\mathcal{S}$ over all $u^\alpha$, $v^\alpha$, and $a_\alpha$, given the specified value of $m$. The solution to this minimization problem is known from linear algebra. We now think of $\psi_{ij}$ as a rectangular matrix. The solution is produced by the singular value decomposition[13] of $\psi$,
$$
\psi = UDV^T, \tag{11}
$$
where $U$ and $D$ are $\ell \times \ell$ matrices, $V$ is an $\ell \times J$ matrix (where $j = 1,\ldots,J$, and we assume $J \geq \ell$), $U$ is orthogonal, $V$ is column orthogonal, and the diagonal matrix $D$ contains the singular values of $\psi$. Linear algebra tells us that the $u^\alpha$, $v^\alpha$, and $a_\alpha$ which minimize $\mathcal{S}$ are given as follows: the $m$ largest-magnitude diagonal elements of $D$ are the $a_\alpha$ and the corresponding columns of $U$ and $V$ are the $u$ and $v$. (We emphasize that the singular value decomposition is not being used here as a numerical method, only as a convenient factorization which allows us to use a theoretical result from linear algebra.)

These optimal states $u^\alpha$ are also eigenvectors of the reduced density matrix of the system as part of the universe. This reduced density matrix for the system depends on the state of the universe, which in this case is a pure state $|\psi\rangle$. [The universe could also be in a mixed state (see below) or at finite temperature.] The density matrix for the system in this case is given by
$$
\rho_{ii'} = \sum_j \psi_{ij} \psi_{i'j}. \tag{12}
$$
We see that
$$
\rho = UD^2U^T; \tag{13}
$$
i.e., $U$ diagonalizes $\rho$. The eigenvalues of $\rho$ are $w_\alpha = a_\alpha^2$ and the optimal states $u$ are the eigenvectors of $\rho$ with the largest eigenvalues. Each $w_\alpha$ represents the probability of the system being in the state $u^\alpha$, with $\sum_\alpha w_\alpha = 1$. The deviation of $P_m \equiv \sum_{\alpha=1}^m w_\alpha$ from unity measures the accuracy of the truncation to $m$ states.

To summarize, in the previous two paragraphs we have shown that when the entire lattice is assumed to be in a pure state, the optimal states to keep are the $m$ most significant eigenstates of the reduced density matrix of the block $BB$, obtained from the wave function of the entire lattice via Eq. (12).

We can also consider the universe to be in a mixed state. This is the natural assumption for a system at finite temperature, and it is also useful to assume a mixed

---

<!-- Page 5 -->

48
DENSITY-MATRIX ALGORITHMS FOR QUANTUM . . .
10 349

state when one wishes to obtain several of the lowest-lying states: If we put the lattice with equal probability into each of several states, then the block states obtained from the density matrix will equally well represent each of these lattice states. We represent the mixed case by saying that the lattice has probability $W_k$ to be in state $|\psi^k\rangle$. If the system is at a finite temperature, then the $W_k$ are normalized Boltzmann weights. In this case the appropriate definition for the error in the representation is
$$
\mathcal{S} = \sum_k W_k \sum_{ij} \left( \psi_{ij}^k - \sum_{\alpha=1}^m a_{\alpha}^k u_i^{\alpha} v_j^{k,\alpha} \right)^2. \tag{14}
$$
Note that we are interested in determining a single set of optimal $u^{\alpha}$, whereas we allow the rest of the universe additional freedom to choose a different $v^{\alpha}$ for each state $k$. Minimizing over the $u^{\alpha}$, $v^{k,\alpha}$, and $a_{\alpha}^k$, we find
$$
\rho u^{\alpha} = w_{\alpha} u^{\alpha}, \tag{15}
$$
with
$$
\rho_{i,i'} = \sum_k W_k \sum_j \psi_{ij}^k \psi_{i'j}^k \tag{16}
$$
and
$$
w_{\alpha} = \sum_k W_k (a_{\alpha}^k)^2. \tag{17}
$$
This equation for $\rho$ is the definition of the reduced density matrix when the universe is in a mixed state, and the $u^{\alpha}$ are the eigenstates of $\rho$.

Thus the conclusion when the universe is in a mixed state is identical to the result for a pure state: The optimal states to keep are the eigenvectors of the reduced density matrix with the largest eigenvalues.

## V. DENSITY-MATRIX ALGORITHMS

Incorporating the result of the previous section in a numerical renormalization group algorithm involves a fundamental change in the way the calculation is carried out. In the standard RG approach, to find the states to be kept, one diagonalizes only the system $BB$, which becomes $B'$. In the density-matrix approach, in order to obtain any reasonable approximation to the density matrix, it is necessary to diagonalize the Hamiltonian of a larger system which includes $BB$, namely, some sort of superblock, and then use the eigenstates of the superblock to determine the density matrix. The density matrix is then diagonalized, and its most significant eigenstates are the states kept. The number of eigenstates of the Hamiltonian of the superblock used to produce the density matrix can be as small as 1; this single state produces a density matrix for $BB$ which has many eigenstates to be used as block states to be kept.

A density-matrix algorithm is defined mainly by the form of the superblock and the manner in which the blocks are enlarged (such as by doubling the block, $B' = BB$, or by adding a single site, $B' = B + \text{site}$), and by the choice of superblock eigenstates used in constructing the density matrix (e.g., the two lowest-lying $S_z = 0$ states). An eigenstate of the superblock Hamiltonian is called a target state if it is used in forming the block density matrix. The most efficient algorithms use only a single target state (usually the ground state) in constructing the density matrix. By targeting only one state, the block states are more specialized for representing that state, and fewer are needed for a given accuracy. Probably the most important characteristic of a density-matrix algorithm is the rate at which the accuracy increases with the number of states $m$. We have found that the accuracy of the representation of the target states increases roughly exponentially with $m$, at least for open boundary conditions. The coefficient governing the increase of accuracy with $m$ is largest with a single target state.

Several considerations enter in the construction of the superblock to be used in an algorithm. Generally it is more efficient to enlarge the block by adding a single site, rather than doubling a block, for the same reasons as in the standard RG approach (see Sec. II). A block can be represented more accurately with a given $m$ if it connects to the rest of the chain only on one end, rather than both ends. A block can connect to the rest of the chain by only one end if it is on an end of a chain with open boundary conditions. If periodic boundary conditions are used, it is not possible for a block to connect on only one end. Below we give results of calculations which show that the open boundary condition case performs much better than the periodic boundary condition case. Our intuitive picture for this numerical result is the following: Roughly speaking, each eigenstate of the block density matrix represents the response of the block to a particular quantum fluctuation in the rest of the chain. A block with two ends which connect to the rest of the system must respond to nearly independent fluctuations near each of the ends. In the case of a long block, where the ends are nearly independent, if $m$ states are required to accurately describe a single end to a given accuracy, then approximately $m^2$ states would be required to accurately represent both ends. In other words, if for a given accuracy open boundary condition require $m$ states, periodic boundary conditions require roughly $m^2$ states.

Figure 1 shows the superblock configuration used for most of the calculations reported here. We adopt the notation $B_{\ell} \bullet \bullet B_{\ell'}^R$ for this configuration, where $B_{\ell}$ represents a block composed of $\ell$ sites, $B_{\ell'}^R$ is a reflected block (right interchanged with left) of length $\ell'$, $\bullet$ represents a single site, and the total length of the superblock is $L = \ell + \ell' + 2$. Here $B' = B_{\ell+1}$ is formed from the left block plus a single site, i.e., $B_{\ell+1} = B_{\ell} \bullet$. Open boundary conditions are used. The right-hand block and site $\bullet B_{\ell'}^R$

**[Figure 1: The configuration of blocks used for the density matrix calculations considered here.]**

> **Figure Structure:** Single diagram
>
> **--- PANEL A: Superblock Configuration Diagram ---**
>
> **Axes:**
> - **Bottom Axis (X):** Not applicable (diagram, not numerical plot)
> - **Top Axis:** Not applicable
> - **Left Axis (Y):** Not applicable
> - **Right Axis:** Not applicable
>
> **Data Series/Curves:**
> - **Blocks:** Rectangles representing blocks containing $\ell$ and $\ell'$ sites
> - **Sites:** Solid circles representing single sites
>
> **Key Data Points:**
> - $B_{\ell}$: Block of length $\ell$ (left rectangle)
> - $B_{\ell'}^R$: Reflected block of length $\ell'$ (right rectangle)
> - $\bullet$: Single site (two solid circles between blocks)
> - $B_{\ell+1}'$: New block formed by adding site to left block
>
> **Annotations:**
> - $B_{\ell}$: Label for left block
> - $B_{\ell'}^R$: Label for right block
> - $B_{\ell+1}'$: Label for new block formed from left block plus single site
> - Curved arrow showing formation of $B_{\ell+1}'$
>
> **Legend:** Not applicable (simple diagram with labeled components)
>
> **Panel Description:** Diagram showing superblock configuration with two blocks ($B_{\ell}$ and $B_{\ell'}^R$) connected by two single sites ($\bullet$). The left block $B_{\ell}$ is shown combining with one site to form $B_{\ell+1}'$.
>
> **--- END PANEL ---**
>
> **Global Figure Description:** This diagram illustrates the superblock configuration used in density matrix calculations, where the superblock consists of a left block $B_{\ell}$, two single sites, and a right reflected block $B_{\ell'}^R$. The configuration demonstrates how the block is enlarged by adding a single site to form $B_{\ell+1}'$.
> **Additional Details:** The diagram shows the block enlargement process where $B_{\ell+1}' = B_{\ell} \bullet$, with the right-hand block and site $\bullet B_{\ell'}^R$ also shown. The total length of the superblock is $L = \ell + \ell' + 2$.

---

<!-- Page 6 -->

10 350
STEVEN R. WHITE
48

are only used to help form the density matrix for $B_{\ell+1}$; in the construction of the density matrix, the states of $\bullet B_{\ell'}^R$ are traced over. This configuration can be used in two different ways: in an infinite chain method, in which the chain size increases by two at each step, and in a finite chain method, in which the chain size is fixed.

## A. Infinite system method

In the first step of the infinite system method, we start with a four site chain and diagonalize the Hamiltonian of the superblock configuration $B_1 \bullet \bullet B_1^R$, where $B_1$ and $B_1^R$ both represent a single site. We use the Davidson algorithm$[14]$ for the sparse matrix diagonalization, but one could also use the more well-known Lanczos method. Using the target states calculated with this configuration, we calculate a density matrix and form an effective Hamiltonian for $B_2 = B_1 \bullet$. In the second step we diagonalize $B_2 \bullet \bullet B_2^R$, where we have formed $B_2^R$ by reflecting $B_2$. We continue in this manner, diagonalizing the configuration $B_{\ell} \bullet \bullet B_{\ell}^R$, and setting $B_{\ell+1} = B_{\ell} \bullet$, and using $B_{\ell+1}$ and its reflection in the next step of the iteration. At each step, both blocks increase in length by one site, and the total length of the chain increases by 2 at each step of the iteration. The infinite chain method is usually used when one is interested in ground state properties of the infinite chain. Each step of the iteration pushes the ends of the chain farther from the two sites in the center. After many steps, each block approximately represents one-half of an infinite chain, $B$ must not only contain many sites itself; its effective Hamiltonian must be formed from a system in which the rest of the chain has many sites. The effective Hamiltonian formed from the left-hand side $B_{\ell} \bullet$ depends strongly on the right-hand side $\bullet B_{\ell}^R$. The infinite chain algorithm converges in two senses simultaneously: in the length of $B_{\ell}$ going to infinity and in the sense that $B_{\ell}$ is adapted to respond to an infinite chain connected to it on the right.

Why not use the simpler configuration $B_{\ell} \bullet B_{\ell}^R$? In other words, why make the right-hand side $\bullet B_{\ell}^R$ rather than $B_{\ell}^R$? If there is only one target state, and $B_{\ell}^R$ has $m$ states, then from Eq. (12), the density matrix $\rho$ for the system $B_{\ell} \bullet$ has at most $m$ nonzero eigenvalues. Initially, when $B_{\ell}^R$ (and $B_{\ell}$) consists of just a single site, it has only a few states. With this method, the number of states in $B_{\ell+1}$ could not be larger than the number in $B_{\ell}^R$ (which is the same as the number in $B_{\ell}$), unless one included states with density-matrix eigenvalues of zero. In general, for a robust method, the number of nonzero eigenvalues of the density matrix should be larger than the number of states kept, except in the first few steps when all states of the block are kept. If more than one state is targeted, then the $B_{\ell} \bullet B_{\ell}^R$ configuration is presumably feasible, although we have not tested it. In any case, the presence of extra sites in the center which are not part of the outer blocks makes the method more accurate, since $B_{\ell}^R$ is represented by an approximate Hamiltonian, whereas the central sites are represented exactly. Thus extra sites in the center produce a more accurate density matrix. The penalty for these extra sites is that the diagonalization of the superblock is more work.

The infinite system algorithm is summarized in Table II. The representation of the blocks is identical to that of the standard algorithm: We describe a block by listing how many states it has and the quantum numbers for each state, and by storing matrices for $H_B$, $S_i^z$, etc. Once the matrix $O$ is constructed using the most significant eigenvectors of the density matrix, the change of basis procedure it also identical to that of the standard algorithm. For the purposes of organizing the algorithm, it is easiest to think of the two sites in the middle as blocks which can be treated similarly to the two outer blocks, although they contain only a few states.

## B. Finite system method

The finite system algorithm is designed to calculate accurately the properties of a finite system of size $L$, which we will assume for simplicity to be even. It is summarized in Table III. It begins with the use of the infinite system algorithm for $L/2 - 1$ steps, so that the final superblock used is of size $L$. In the infinite system method, there is no need to store $B_{\ell}$ once we have $B_{\ell+1}$; we need only

TABLE II. Infinite system density-matrix algorithm for a 1D system.

| Step | Description                                                                                                                                                                                                                                                                                             |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.   | Make four initial blocks, each consisting of a single site, representing the initial four site system. Set up matrices representing the block Hamiltonian and other operators.                                                                                                                          |
| 2.   | Form the Hamiltonian matrix (in sparse form) for the superblock.                                                                                                                                                                                                                                        |
| 3.   | Using the Davidson or Lanczos method, diagonalize the superblock Hamiltonian to find the target state $\psi(i_1,i_2,i_3,i_4)$. $\psi$ is usually the ground state. Expectation values of various operators can be measured at this point using $\psi$.                                                  |
| 4.   | Form the reduced density matrix for the two-block system 1-2, using $\rho(i_1,i_2;i_1',i_2') = \sum_{i_3,i_4} \psi(i_1,i_2,i_3,i_4)\psi(i_1',i_2',i_3,i_4)$.                                                                                                                                            |
| 5.   | Diagonalize $\rho$ to find a set of eigenvalues $w_\alpha$ and eigenvectors $u_{i_1,i_2}^\alpha$. Discard all but the largest $m$ eigenvalues and associated eigenvectors.                                                                                                                              |
| 6.   | Form matrix representations of operators (such as $H$) for the two-block system 1-2 from operators for each separate block [cf. Eq. (4)].                                                                                                                                                               |
| 7.   | Form a new block 1 by changing basis to the $u^\alpha$ and truncating to $m$ states using $H^{1'} = OH^{12}O^\dagger$, etc. If blocks 1 and 2 have $m_1$ and $m_2$ states, then $O$ is an $m \times m_1m_2$ matrix, with matrix elements $O(\alpha;i_1,i_2) = u_{i_1,i_2}^\alpha, \alpha = 1,\ldots,m$. |
| 8.   | Replace old block 1 with new block 1.                                                                                                                                                                                                                                                                   |
| 9.   | Replace old block 4 with the reflection of new block 1.                                                                                                                                                                                                                                                 |
| 10.  | Go to step 2.                                                                                                                                                                                                                                                                                           |

---

<!-- Page 7 -->

48 DENSITY-MATRIX ALGORITHMS FOR QUANTUM ... 10 351

TABLE III. Finite system density-matrix algorithm for a 1D system consisting of $L$ sites. A calculation consists of several iterations, indexed by $I$, with each iteration consisting of $L-3$ steps, indexed by $\ell$, where $\ell$ is the size of the first block.

| Step | Description                                                                                                                                                                                                                                                                  |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.   | (First half of $I = 1$.) Use the infinite system algorithm for $L/2 - 1$ steps to build up the lattice to $L$ sites. At each iteration store the block Hamiltonian and end operator matrices for block 1. Label the blocks by their size, $B_\ell$, $\ell = 1, \ldots, L/2$. |
| 2.   | (Start of second half of $I = 1$.) Set $\ell = L/2$. Use $B_\ell$ as block 1, and the reflection of $B_{L-\ell-2}$ as block 4.                                                                                                                                               |
| 3.   | Steps 2-8 of Table II.                                                                                                                                                                                                                                                       |
| 4.   | Store the new block 1 as $B_{\ell+1}$, replacing the old $B_{\ell+1}$.                                                                                                                                                                                                       |
| 5.   | Replace block 4 with the reflection of $B_{L-\ell-2}$, obtained from the first half of this iteration.                                                                                                                                                                       |
| 6.   | If $\ell < L-3$, set $\ell = \ell + 1$ and go to step 3.                                                                                                                                                                                                                     |
| 7.   | (Start of iteration $I$, $I > 2$.) Make four initial blocks, the first three consisting of a single site, and the fourth consisting of the reflection of $B_{L-3}$ from the previous iteration. Set $\ell = 1$.                                                              |
| 8.   | Steps 2-8 of Table II.                                                                                                                                                                                                                                                       |
| 9.   | Store the new block 1 as $B_{\ell+1}$, replacing the old $B_{\ell+1}$.                                                                                                                                                                                                       |
| 10.  | Replace block 4 with the reflection of $B_{L-\ell-2}$, obtained from the previous iteration (if $\ell \leq L/2 - 1$) or the first half of this iteration (if $\ell > L/2 - 1$).                                                                                              |
| 11.  | If $\ell < L-3$, set $\ell = \ell + 1$ and go to step 8. If $\ell = L-3$, start a new iteration by going to step 7. (Stop after 2 or 3 iterations.)                                                                                                                          |

store the latest block. In the finite system method, we need to store $L-3$ blocks, $B_1$ to $B_{L-3}$, and the infinite system method is used to get initial, approximate versions of $B_1$ to $B_{L/2}$. After the system $B_{L/2-1} \bullet \bullet B_{L/2-1}^R$ is used to form $B_{L/2}$, the next step is to use the configuration $B_{L/2} \bullet \bullet B_{L/2-2}^R$ to form $B_{L/2+1}$. This system, and all the other superblocks to follow, contains $L$ sites. We continue to form the other blocks up to size $L-3$, using the superblock $B_\ell \bullet \bullet B_{L-\ell-2}^R$ to form $B_{\ell+1}$. This sequence of steps is the first iteration of the finite system algorithm.

The second and subsequent iterations use the blocks obtained from the previous iteration as the right-hand reflected blocks in each superblock. The first step starts by diagonalizing the superblock $B_1 \bullet \bullet B_{L-3}^R$, where $B_1$ is a single site and is always known exactly, and $B_{L-3}^R$ is obtained from the last step of the previous iteration. Once a new $B_\ell$ is formed, it replaces the old $B_\ell$, so that only one set of blocks need be stored. Consequently, for the second half of the iteration, starting with the superblock $B_{L/2-1} \bullet \bullet B_{L/2-1}^R$, we use a block formed in the current iteration, rather than the last iteration, as the right-hand block. On the very last iteration, we usually stop after the diagonalization of $B_{L/2-1} \bullet \bullet B_{L/2-1}^R$, and then use this wave function of the $L$-site system to measure various properties, such as the local magnetization or correlation functions.

After a few iterations each $B_\ell$ accurately represents an $\ell$-site block which is the left-hand $\ell$ sites of an $L$-site chain. Usually the method converges by the middle of the second iteration, although sometimes three iterations are necessary.

An important improvement in efficiency can be made by keeping a different number of states $m'$ in the right-hand block from the number in the left-hand block $m$. Fewer states are needed in the right-hand block because this block is only used to help produce the density matrix, whereas the left-hand block not only is used to produce the density matrix; it is part of the new block we are producing. Thus the left-hand block is more important, and we should take $m > m'$. This can be done quite simply by keeping only $m'$ states in both blocks in all but the last iteration, and in the last iteration keeping $m$ states for the left block. Assuming we stop after diagonalizing the $B_{L/2-1} \bullet \bullet B_{L/2-1}^R$ system, then this very last diagonalization will have $m$ states for both blocks, and it will be especially accurate. Typically we take $m/m' = 2$ or 3. If only one state is targeted, then one should not try to make $m > nm'$, where $n$ is the number of states on a single site, since the number of nonzero eigenvalues of the density matrix cannot be more than $nm'$ times the number of states targeted. We find that in typical cases, the accuracy of the calculation for fixed $m$ is approximately the same using this procedure for any $m'$ ranging from $m' \approx m/2$ to $m' = m$. Of course, the calculation time and storage is substantially less for smaller values of $m'$.

### C. Periodic boundary conditions

To perform calculations with periodic boundary conditions, a slight variation of the finite system method can be used. In this case it is most convenient to use the superblock configuration $B_\ell \bullet B_{\ell'}^R \bullet$, with $B_{\ell+1}' = B_\ell \bullet$. This configuration is preferred over $B_\ell \bullet \bullet B_{\ell'}^R$ because it does not have the two big blocks as neighbors. If the two big blocks are neighbors, the sparseness of the Hamiltonian (measured by the average number of nonzero matrix elements per row of the matrix) for the superblock is greatly reduced (meaning more elements are nonzero). The off-diagonal terms of the Hamiltonian matrix come from operators such as $S_r^+ S_\ell^-$, where $r$ is the right-hand site of a block, say block 1, and $\ell$ is the left-hand site of block 2. The term in the Hamiltonian for this operator is of the form

$$
[S_r^+]_{i_1i_1'}[S_\ell^-]_{i_2i_2'}\delta_{i_3i_3'}\delta_{i_4i_4'} \tag{18}
$$

The sparseness of this term is measured by the number of nonzero terms as we vary $i_1'$ and $i_2'$, keeping $i_1$ and $i_2$ fixed. If both block 1 and block 2 are blocks containing $m$ states, the number of nonzero elements per row of

---

<!-- Page 8 -->

10 352
STEVEN R. WHITE
48

the Hamiltonian matrix is proportional to $m^2$, whereas if block 2 is a single site, the number is proportional to $m$. In the case of periodic boundary conditions, these considerations are relevant because block 4 is connected to block 1, and thus one of these should be a single site. Otherwise, the periodic case is nearly identical to the open boundary case discussed above.

## VI. MEASUREMENTS

Properties of the $L$-site system can be obtained from the wave functions of any of the $L-3$ superblock configurations, although we find that the symmetric configuration (with both the left and right blocks of size $L/2-1$) usually gives the most accurate results. The procedure is to use the wave function $|\psi\rangle$ resulting from the diagonalization of the $L$-site system to evaluate expectation values of the form $\langle \psi | A | \psi \rangle$. Rather complicated operators can be evaluated fairly easily, but dynamical information is not easily obtained. In order to measure $A$, one must have kept operator matrices for the components of $A$. For example, to measure the on-site spin density $S_j^z$ for all sites $j$, one must keep track of matrices $[S_j^z]_{i_1 i_1'}$, for all sites in block 1, and similarly for blocks 2, 3, and 4 (regarding the single sites in the middle as blocks 2 and 3). At each step of each iteration, these operators for blocks 1 and 4 must be updated using analogues of Eqs. (4--6). One then obtains the expectation value using
$$
\langle \psi | S_j^z | \psi \rangle = \sum_{i_1 i_2 i_3 i_4 i_1'} \psi_{i_1 i_2 i_3 i_4}^* [S_j^z]_{i_1 i_1'} \psi_{i_1' i_2 i_3 i_4}, \tag{19}
$$
etc. This procedure gives exact evaluations of $\langle \psi | A | \psi \rangle$ for the approximate eigenstate $|\psi\rangle$.

For a correlation function such as $\langle \psi | S_j^z S_k^z | \psi \rangle$, the evaluation depends on whether $j$ and $k$ are on the same block or not. If they are on different blocks, say, block 1 and block 4, then one need only have kept track of $[S_j^z]_{i_1 i_1'}$ and $[S_k^z]_{i_4 i_4'}$, and one has
$$
\langle \psi | S_j^z S_k^z | \psi \rangle = \sum_{i_1 i_2 i_3 i_4 i_1' i_4'} \psi_{i_1 i_2 i_3 i_4}^* [S_j^z]_{i_1 i_1'} [S_k^z]_{i_4 i_4'} \psi_{i_1' i_2 i_3 i_4'}. \tag{20}
$$
If $j$ and $k$ are on the same block, one should not use
$$
\langle \psi | S_j^z S_k^z | \psi \rangle \approx \sum_{i_1 i_2 i_3 i_4 i_1' i_1''} \psi_{i_1 i_2 i_3 i_4}^* [S_j^z]_{i_1 i_1'} [S_k^z]_{i_1' i_1''} \psi_{i_1'' i_2 i_3 i_4}. \tag{21}
$$
This expression does not evaluate the correlation function exactly within the approximate state $|\psi\rangle$. The sum over $i_1'$ should run over a complete set of states, but does not, whereas the sums over the other variables need run only over those states needed to represent $|\psi\rangle$, since they appear as a subscript in either the $|\psi\rangle$ on the left or on the right. To evaluate this type of correlation function, one needs to have kept track of $[S_j^z S_k^z]_{i_1 i_1'}$ throughout the calculation. One then evaluates
$$
\langle \psi | S_j^z S_k^z | \psi \rangle = \sum_{i_1 i_2 i_3 i_4 i_1'} \psi_{i_1 i_2 i_3 i_4}^* [S_j^z S_k^z]_{i_1 i_1'} \psi_{i_1' i_2 i_3 i_4}. \tag{22}
$$
It is usually more convenient to choose points $i$ and $j$ on different blocks, rather than keep track of complicated operators such as $[S_j^z S_k^z]_{i_1 i_1'}$. A convenient way to calculate a correlation function such as this as a function of $j-k$ is to always put $j$ and $k$ at the same distance (within a lattice spacing) from the center of the chain. As $j-k$ is increased, both points move outwards symmetrically towards the ends of the chain.

## VII. RESULTS

In this section we give a brief survey of results, with the intention of illustrating features of the algorithms rather than giving a study of the model systems. Figure 2 shows the density-matrix eigenvalues $w_\alpha$ for a 32 site system for both open and periodic boundary conditions, and for both $S = 1/2$ and $S = 1$, targeting one state. The figure shows the eigenvalues only for one particular size of block 1, but similar results are obtained at other steps in the iteration. The falloff of $w_\alpha$ is most rapid for the open $S = 1/2$ case. The figure shows that it is possible to obtain an accuracy (truncation error) of better than $10^{-7}$ in this case, keeping only $m = 20$ states. The $S = 1$ open case shows a slightly slower falloff, with an accuracy of $10^{-7}$ obtained with $m = 48$. The periodic cases fall off more slowly than either of the open cases. Keeping $m = 50$ states yields an accuracy of roughly $10^{-5}$--$10^{-6}$. Even more accurate treatment of periodic systems is quite feasible; we have been able to keep as many as $m = 200$ states in a calculation of a 60 site system. Degeneracies in the $w_\alpha$ are clearly visible, which come from spin symmetries. One must adjust $m$ so as not to split a degenerate set of levels, since this will destroy spin symmetries that would otherwise be preserved.

In Fig. 3 we show the relative accuracy of the ground state energy of a 28 site $S = 1/2$ system as a function of

**[Figure 2: Density-matrix eigenvalues $w_\alpha$ vs eigenvalue index $\alpha$ for a $32$ site system for both periodic and open boundary conditions, $S = 1/2$ and $S = 1$.]**

> **Figure Structure:** Single plot
>
> **Axes:**
> - **Bottom Axis (X):** $\alpha$ (eigenvalue index), linear scale, range $0$ to $50$
> - **Left Axis (Y):** $w_\alpha$ (density-matrix eigenvalues), logarithmic scale, range $10^0$ to $10^{-7}$
>
> **Data Series/Curves:**
> - Periodic, S=1: open squares, solid line
> - Periodic, S=1/2: asterisk, solid line
> - Open, S=1: filled inverted triangles, solid line
> - Open, S=1/2: filled triangles, solid line
>
> **Key Data Points:**
> - Steplike structure visible in all curves
> - rapid falloff for open cases
> - slower falloff for periodic cases
>
> **Annotations:**
> - "L=32" in top right corner
> - "Periodic, S=1", "Periodic, S=1/2", "Open, S=1" and "Open, S=1/2" in legend
>
> **Legend:**
> - $\square$ Periodic, S=1
> - $\ast$ Periodic, S=1/2
> - $\blacktriangledown$ Open, S=1
> - $\blacktriangle$ Open, S=1/2
>
> **Panel Description:** Plot showing density-matrix eigenvalues $w_\alpha$ versus eigenvalue index $\alpha$ for a 32 site system under different boundary conditions (periodic/open) and spin values (S=1/2, S=1)
>
> **--- END PANEL ---**
>
> **Global Figure Description:** Comparison of density-matrix eigenvalue decay for different system configurations. The plot demonstrates that the eigenvalues fall off more rapidly for open boundary conditions compared to periodic boundary conditions, and for S=1/2 compared to S=1.
> **Additional Details:** The eigenvalues were obtained from the $B_{15} \bullet \bullet B_{15}^R$ system. The steplike structure comes from  the presence of spin degeneracies. Additional near degeneracies in the S=1 cases come from the presence of nearly free S=1/2 effective spins on the ends of S=1 blocks. The errors in the calculation is determined by $\sum_{\alpha=m+1}^{\infty}$, which can be estimated from the figure.

---

<!-- Page 9 -->

48
DENSITY-MATRIX ALGORITHMS FOR QUANTUM ...
10 353

**[Figure 3: Relative error in the ground state energy $\Delta E/E$ for a $28$ site $S = 1/2$ system as a function of the number of states kept $m$, for periodic and open boundary conditions. The exact energy was obtained from Ref. 16 for the periodic case, and from the renormalization group calculation itself with $m = 58$ for the open case.]**

> **Figure Structure:** Single plot
>
> **Axes:**
> - **Bottom Axis (X):** $m$, linear scale, range $0$ to $125$
> - **Left Axis (Y):** $\Delta E/|E|$, logarithmic scale, range $10^{-1}$ to $10^{-9}$
>
> **Data Series/Curves:**
> - **Periodic BCs:** open squares, solid line
> - **Open BCs:** filled circles, solid line
>
> **Key Data Points:**
> - Periodic BCs: error decreases steadily with increasing $m$
> - Open BCs: error decreases more rapidly than periodic BCs
> - At $m=40$, Open BCs error $< 10^{-9}$
> - At $m=100$, Periodic BCs error $\approx 10^{-6}$
>
> **Annotations:**
> - "S=1/2, L=28" in top right corner
> - "Periodic BCs" and "Open BCs" in legend
>
> **Legend:**
> - $\square$ Periodic BCs
> - $\bullet$ Open BCs
>
> **Panel Description:** Plot showing the relative error in ground state energy as a function of the number of states kept $m$ for a 28-site $S=1/2$ system with two different boundary conditions.
>
> **Global Figure Description:** The figure compares the convergence of the density-matrix algorithm for periodic and open boundary conditions in a 28-site $S=1/2$ system. The open boundary condition case converges more rapidly than the periodic case.
> **Additional Details:** Both axes use linear scale for the x-axis and logarithmic scale for the y-axis.

the number of states kept. The error for the periodic case was obtained from comparison with an exact diagonalization calculation (the ground-state energy per spin is $E = -0.4442017$);[16] for the open case, we used the renormalization group calculation itself with $m = 60$, which we believe reproduces the ground state energy to better than $10^{-10}$. (For smaller systems, we have verified that our open case really is this accurate by comparison with our own exact diagonalization calculations.) Again, we see that the open boundary condition case is extremely accurate even for a relatively small $m$. The relative error in the energy is somewhat larger than the truncation error indicated by the density-matrix eigenvalues. The fall-off in the error in the energy is much slower for the periodic case; it is difficult to achieve an accuracy of better than $10^{-7}$ in this case. Figure 4 shows similar results for a 16 site $S = 1$ system, which is close to the largest system currently feasible with exact diagonalization. As in the $S = 1/2$ case, the error for the periodic case was obtained from comparison with an exact diagonalization calculation (the ground-state energy per unit site is $E = -2.80585/2$),[17] and from the renormalization group calculation itself with $m = 140$ for the open case. The accuracy for $S = 1$ case shows the same overall behavior as the $S = 1/2$ case, although it is slightly less accurate.

Using periodic boundary conditions, we calculated the gap $\Delta_L$ between the ground state, which has $S_z^{\text{total}} = 0$, and the first excited state, which has $S_z^{\text{total}} = 1$, as a function of lattice size, Fig. 5. For the $S = 1/2$ systems we used $m = 80$, and for the largest $S = 1$ systems we used $m = 200$. The gap for the $S = 1$ system clearly tends to a finite value as $L \to \infty$, in agreement with Haldane's conjecture that integral-spin Heisenberg chains are gapped. The gap in the $S = 1/2$ system tends to 0 as $L \to \infty$, in agreement with the Bethe ansatz exact solution of the model. These results predict an infinite system gap for the $S = 1$ chain of $0.411(1)$. Using open boundary conditions (of the "soft" type discussed below) we have obtained what we believe is the most accurate value of $\Delta$ to date: $\Delta = 0.4105(1)$.[15]

It is clearly more accurate to use open boundary conditions than periodic in these calculations, but a bulk system is generally better described using periodic boundary conditions. It is important to understand the effect of an open boundary on these systems. The effect of an open boundary is most easily seen for an $S = 1/2$ system by measuring the local bond strength $\langle \mathbf{S}_j \cdot \mathbf{S}_{j+1} \rangle$ as a function of site index $j$. In Fig. 6(a) we show the local bond strength for a 60 site system. The open boundaries cause a strong alternation in the bond strength which decays very slowly. (We emphasize that this is the effect of the boundaries; errors in our calculation are negligible.) This effect is easily understood using the valence bond picture of the ground state of the $S = 1/2$ system. In the valence bond picture, the ground state is viewed as a resonance between two different states, one having strong singlet bonds on the even numbered links of the chain, and no bonds on the odd links, and the other with strong bonds on the odd links, and none on the even.

**[Figure 4: Relative error in the ground state energy $E$ for a $16$ site $S = 1$ system, as a function of the number of states kept $m$, for periodic and open boundary conditions. The exact energy was obtained from Ref. 17 for the periodic case, and from the renormalization group calculation itself with $m = 140$ for the open case.]**

> **Figure Structure:** Single plot
>
> **Axes:**
> - **Bottom Axis (X):** $m$, linear scale, range $0$ to $125$
> - **Left Axis (Y):** $\Delta E/|E|$, logarithmic scale, range $10^{-1}$ to $10^{-9}$
>
> **Data Series/Curves:**
> - **Periodic BCs:** open squares, solid line
> - **Open BCs:** filled circles, solid line
>
> **Key Data Points:**
> - Periodic BCs: error decreases steadily with increasing $m$
> - Open BCs: error decreases more rapidly than periodic BCs
> - At $m=80$, Open BCs error $< 10^{-9}$
> - At $m=125$, Periodic BCs error $\approx 10^{-6}$
>
> **Annotations:**
> - "S=1, L=16" in top right corner
> - "Periodic BCs" and "Open BCs" in legend
>
> **Legend:**
> - $\square$ Periodic BCs
> - $\bullet$ Open BCs
>
> **Panel Description:** Plot showing the relative error in ground state energy as a function of the number of states kept $m$ for a 16-site $S=1$ system with two different boundary conditions.
>
> **Global Figure Description:** The figure compares the convergence of the density-matrix algorithm for periodic and open boundary conditions in a 16-site $S=1$ system. The open boundary condition case converges more rapidly than the periodic case, similar to Figure 3.
> **Additional Details:** Both axes use linear scale for the x-axis and logarithmic scale for the y-axis.

**[Figure 5: Gaps $\Delta_L$ between the ground and first excited states for $S = 1/2$ and $S = 1$ systems with periodic boundary conditions versus the inverse length of the chain $1/L$.]**

> **Figure Structure:** Single plot
>
> **Axes:**
> - **Bottom Axis (X):** $1/L$, linear scale, range $0.00$ to $0.10$
> - **Left Axis (Y):** $\Delta_L$, linear scale, range $0.0$ to $0.5$
>
> **Data Series/Curves:**
> - **S=1:** open circles, solid line
> - **S=1/2:** filled circles, solid line
>
> **Key Data Points:**
> - S=1: gap tends to finite value as $1/L \to 0$ (i.e., $L \to \infty$)
> - S=1/2: gap tends to zero as $1/L \to 0$
> - For S=1 at $1/L=0.083$, $\Delta_L \approx 0.5$
> - For S=1/2 at $1/L=0.083$, $\Delta_L \approx 0.35$
>
> **Annotations:**
> - "S=1" and "S=1/2" in legend
>
> **Legend:**
> - $\circ$ S=1
> - $\bullet$ S=1/2
>
> **Panel Description:** Plot showing the energy gap $\Delta_L$ between ground and first excited states as a function of inverse chain length $1/L$ for two different spin systems.
>
> **Global Figure Description:** The figure demonstrates the different scaling behavior of the energy gap for $S=1/2$ and $S=1$ systems. The $S=1$ system shows a finite gap in the thermodynamic limit, consistent with Haldane's conjecture, while the $S=1/2$ system shows a vanishing gap, consistent with the Bethe ansatz solution.
> **Additional Details:** Both axes use linear scale. The data points for S=1 are connected by a solid line, and data points for S=1/2 are connected by a solid line.

---

<!-- Page 10 -->

10 354
STEVEN R. WHITE
48

Simple variational estimates show that this type of state is greatly preferred over a Néel ordered state. This picture strongly suggests that the system can be perturbed very easily into one of these two dimerized states. The open boundary conditions favor a strong bond on the outermost links, since that is the only way the end sites can participate in a bond.

If the system has an odd number of sites, one end favors the strong bonds on the even links, and the other favors them on the odd links; thus the system is frustrated. In Fig. 6(b) we show the local bond strength on a 61 site system. The system shows the odd links with stronger bonds on the left, and the even links stronger on the right. For other system sizes the behavior is nearly identical to that shown for $L = 60$ in the sense that there are never more than a few links in the center with almost no bond-strength alternation present, and there is approximately linear falloff of the bond strength in the central region. We interpret this behavior as evidence of a mobile domain wall (soliton), with a nearly uniform probability to be at any particular site in the central region. If one averages over the various positions of the domain wall, one obtains results consistent with the figure.

If we desire the accuracy of open boundary conditions, but not the bond-strength alternation which is absent in the bulk system, it is possible to considerably weaken the alternation using "soft" boundary conditions. In Fig. 6(c) we show results from an especially simple version of these boundary conditions. In this case we weakened the two outermost exchange constants $J$ from their bulk

**[Figure 6: Local bond strength for (a) $60$ site and (b) $61$ site $S = 1/2$ chains with open boundary conditions. The systems try to have strong single bonds at the outermost links of the chain; this induces a domain wall in the odd-length chain. In (c), the bond-strength alternation is diminished by weakening the local value of the exchange constant $J$ on the outermost links.]**

> **Figure Structure:** Multi-panel (a-c)
>
> **--- PANEL (a): [L=60, S=1/2, Open BCs] ---**
>
> **Axes:**
> - **Bottom Axis (X):** $i$, linear scale, range $0$ to $60$
> - **Left Axis (Y):** $\langle S_i S_{i+1} \rangle$, linear scale, range $-0.7$ to $-0.2$
>
> **Data Series/Curves:**
> - solid dots, solid line
>
> **Key Data Points:**
> - Oscillatory pattern throughout chain
>
> **Annotations:**
> - Label "(a)" in top left
> - "L=60, S=1/2, Open BCs" centered above
>
> **Legend:** None explicitly shown, but data represents local bond strength
>
> **Panel Description:** Shows local bond strength for a 60-site $S=1/2$ chain with open boundary conditions. The plot shows a clear alternation pattern with stronger bonds at the ends of the chain and weaker bonds in the center.
>
> **--- END PANEL ---**
>
> **--- PANEL (b): [L=61, S=1/2, Open BCs] ---**
>
> **Axes:**
> - **Bottom Axis (X):** $i$, linear scale, range $0$ to $60$
> - **Left Axis (Y):** $\langle S_i S_{i+1} \rangle$, linear scale, range $-0.7$ to $-0.2$
>
> **Data Series/Curves:**
> - solid dots, solid line
>
> **Key Data Points:**
> - Oscillatory pattern throughout chain
>
> **Annotations:**
> - Label "(b)" in top left
> - "L=61, S=1/2, Open BCs" centered above
>
> **Legend:** None explicitly shown, but data represents local bond strength
>
> **Panel Description:** Shows local bond strength for a 61-site $S=1/2$ chain with open boundary conditions. The system shows odd links with stronger bonds on the left, and even links stronger on the right, indicating frustration due to odd number of sites.
>
> **--- END PANEL ---**
>
> **--- PANEL (c): [L=60, S=1/2, Open BCs, $J_{1-2} = J_{59-60} = 0.236$] ---**
>
> **Axes:**
> - **Bottom Axis (X):** $i$, linear scale, range $0$ to $60$
> - **Left Axis (Y):** $\langle S_i S_{i+1} \rangle$, linear scale, range $-0.7$ to $-0.2$
>
> **Data Series/Curves:**
> - solid dots, solid line
>
> **Key Data Points:**
> - More uniform bond strength throughout chain
> - Reduced alternation compared to panel (a)
> - Value approximately -0.45 throughout most of chain
>
> **Annotations:**
> - Label "(c)" in top left of plot
> - "L=60, S=1/2, Open BCs" centered above plot
> - "$J_{1-2} = J_{59-60} = 0.236$" below plot
>
> **Legend:** None explicitly shown, but data represents local bond strength
>
> **Panel Description:** Shows local bond strength for a 60-site $S=1/2$ chain with weakened outermost exchange constants. The bond-strength alternation is diminished compared to panel (a).
>
> **--- END PANEL ---**
>
> **Global Figure Description:** Figure 6 demonstrates how boundary conditions affect bond strength alternation in spin chains. Panels (a) and (b) show the natural dimerization pattern that occurs with open boundary conditions, with panel (b) showing the frustration that occurs with an odd number of sites. Panel (c) shows how weakening the outermost exchange constants can reduce the bond-strength alternation.
> **Additional Details:** All plots show data points with error bars and a smooth curve connecting them. The y-axis represents the expectation value of the bond strength $\langle S_i S_{i+1} \rangle$.

**[Figure 7: Local bond strength (a) and local value of $S_z$ (b) for a $S = 1$ system with 60 sites. The $S = 1/2$ effective spins on the ends of are clearly visible in (b), which shows one of the low-lying triplet states just above the ground state.]**

> **Figure Structure:** Multi-panel (a-b)
>
> **--- PANEL (a): [L=60, S=1, Open BCs] ---**
>
> **Axes:**
> - **Bottom Axis (X):** $i$, linear scale, range $0$to $60$
> - **Left Axis (Y):** $\langle S_i S_{i+1} \rangle$, linear scale, range $-1.7$ to $-1.2$
>
> **Data Series/Curves:**
> - solid dots, solid line
>
> **Key Data Points:**
> - Nearly constant value of approximately -1.45 throughout chain
> - Slight variation at ends of chain
>
> **Annotations:**
> - Label "(a)" in top left
> - "L=60, S=1, Open BCs" centered above
>
> **Legend:** None explicitly shown, but data represents local bond strength
>
> **Panel Description:** Shows local bond strength for a 60-site $S=1$ chain with open boundary conditions. The bond strength is nearly uniform throughout the chain, indicating a different behavior than the $S=1/2$ case.
>
> **--- END PANEL ---**
>
> **--- PANEL (b): [L=60, S=1, Open BCs, $S_z^{\text{tot}}=1$] ---**
>
> **Axes:**
> - **Bottom Axis (X):** $i$, linear scale, range $0$ to $60$
> - **Left Axis (Y):** $\langle S_i^z \rangle$, linear scale, range $-0.4$ to $0.6$
>
> **Data Series/Curves:**
> - solid dots, solid line
>
> **Key Data Points:**
> - Values near zero in the middle of chain
> - Clear indication of $S=1/2$ effective spins at the ends
>
> **Annotations:**
> - Label "(b)" in top left
> - "L=60, S=1, Open BCs, S_z^{\text{tot}}=1" centered above
>
> **Legend:** None explicitly shown, but data represents local magnetization
>
> **Panel Description:** Shows local value of $S_z$ for a 60-site $S=1$ chain with $S_z^{\text{tot}}=1$. The plot shows the presence of $S=1/2$ effective spins at the ends of the chain, which is characteristic of the Haldane phase.
>
> **--- END PANEL ---**
>
> **Global Figure Description:** Figure 7 demonstrates the behavior of a $S=1$ spin chain. Panel (a) shows nearly uniform bond strength, while panel (b) shows the characteristic edge states of the Haldane phase with $S=1/2$ effective spins at the ends of the chain.
> **Additional Details:** Both panels show data points with error bars and a smooth curve connecting them. Panel (b) specifically shows the local magnetization $\langle S_i^z \rangle$ which reveals the edge states.

---

<!-- Page 11 -->

48
DENSITY-MATRIX ALGORITHMS FOR QUANTUM...
10 355

**[Figure 8: Density-matrix eigenvalues $w_\alpha$ vs $\alpha$ for a $32$ site $S = 1/2$ system with open boundary conditions, with various numbers of target states.]**

> **Figure Structure:** Single plot
>
> **Axes:**
> - **Bottom Axis (X):** $\alpha$, linear scale, range $0$ to $50$
> - **Left Axis (Y):** $w_\alpha$, logarithmic scale, range $10^0$ to $10^{-8}$
>
> **Data Series/Curves:**
> - 1 Target State: open squares, soild line
> - 2 Target States: open circles, soild line
> - 3 Target States: open triangles, soild line
> - 4 Target States: filled squares, soild line
> - 5 Target States: filled circles, soild line
>
> **Key Data Points:** The eigenvalues decay exponentially with $\alpha$ for all target states. The rate of decay increases with the number of target states.
>
> **Annotations:**
> - "S=1/2, L=32, Open BCs, $S_z^{\text{tot}}=1$" centered above
> - "Target States: 1, 2, 3, 4, 5" in the legend
>
> **Legend:**
> - Target States:
> - $\square$ 1
> - $\circ$ 2
> - $\triangle$ 3
> - $\blacksquare$ 4
> - $\bullet$ 5
>
> **Panel Description:** This plot shows density-matrix eigenvalues $w_\alpha$ versus $\alpha$ for a 32-site $S = 1/2$ system with open boundary conditions, with various numbers of target states.
>
> **Global Figure Description:** The plot demonstrates how the density-matrix eigenvalues decay as a function of $\alpha$ for different numbers of target states. As more target states are used, the eigenvalues decay more rapidly, indicating that more states need to be kept for accurate calculations.
> **Additional Details:** The eigenvalues were obtained from the $B_{15} \bullet \bullet B_{15}^R$ system. When a greater number of states are targeted for a fixed value of $m$, the size of the $w_\alpha$ which are discarded is increased, and there is a reduction in the accuracy.

value $J = 1$. The strength was chosen to minimize the bond-strength alternation in the center. In the limit that these outermost couplings are zero, the even sublattice is favored, whereas in the usual case the odd sublattice is favored. Clearly for some intermediate coupling the alternation is nearly absent; in this case it is for $J = 0.236$. More work needs to be done to understand these soft boundary conditions.

The boundaries of an open system pose much less of a problem for an $S = 1$ system, and in addition, are of substantial interest. Spin-spin correlation functions decay exponentially in this system, and the effect of the open ends is also reduced exponentially as we move towards the interior. Figure 7(a) shows the local bond strength of an $S = 1$ chain. Within the central region of the chain, the bond strength is constant to very high accuracy. The open ends of $S = 1$ chains act as effective spin $1/2$'s. These spin $1/2$'s bind very weakly through the chain to form a singlet and triplet. The singlet is the ground state for even numbered chains, and the triplet is the ground state for odd numbered chains. Figure 7(b) shows the local spin magnetization for one of the triplet states of a 60 site chain, which has the spin $1/2$'s on both ends pointing up. As reported earlier,[3] the magnetization of the ends is $0.532$, as opposed to exactly $1/2$ for a real spin $1/2$.

The renormalization group method is most accurate when only one state is used as a target. Figure 8 shows the behavior of the density matrix eigenvalues $w_\alpha$ of a typical system as the number of target states is varied. It is possible to target 15 or more states and still retain reasonable accuracy for a system with open boundary conditions. In this way it is possible to map out the low-lying elementary excitations of a many-body system.

## VIII. CONCLUSIONS

The density-matrix algorithms for numerical renormalization group calculations of 1D systems are very powerful. They provide extremely accurate results for the ground state and low-lying excited states, and can be used to calculate a variety of static quantities. The methods can be applied to a wide variety of systems, and work is currently underway to apply the methods to strongly interacting fermion systems, such as the Hubbard chain and the 1D Kondo lattice. In this paper we have concentrated on the description of a few algorithms, rather than giving detailed results for Heisenberg chains. We plan to publish a survey of a variety of properties of Heisenberg chains shortly.

These methods are somewhat complex and there are substantial variations possible in the construction of algorithms. More work needs to be done in exploring these variations. For example, one could use a momentum-space basis instead of real-space basis in setting up the calculation. This was recently done without the density-matrix formulation for a $4 \times 4$ Hubbard lattice.[18] Applying the density-matrix formulation in this case would probably substantially improve the calculation. One could apply the real-space method to coupled chains and to two-dimensional systems; one would expect less accuracy than in 1D but one might still obtain valuable results. Other areas for future study include applying magnetic fields to calculate static susceptibilities, studying disordered 1D systems, and studying various types of soft boundary conditions for use with open systems.

## ACKNOWLEDGMENTS

We would like to thank R.M. Noack, D. Huse, C. Yu, I. Affleck, and M.P. Gelfand, for helpful comments and discussion. This work was supported by the Office of Naval Research under Grant No. N00014-91-J-1143 and in part by the University of California through an allocation of computer time on the UC Irvine Convex. Calculations were also performed at SDSC.

## References

1. K.G. Wilson, Rev. Mod. Phys. 47, 773 (1975).
2. S.R. White and R.M. Noack, Phys. Rev. Lett. 68, 3487 (1992).
3. S.R. White, Phys. Rev. Lett. 69, 2863 (1992).
4. J.W. Bray and S.T. Chui, Phys. Rev. B 19, 4876 (1979).
5. C.Y. Pan and Xiyao Chen, Phys. Rev. B 36, 8600 (1987).
6. M.D. Kovarik, Phys. Rev. B 41, 6889 (1990).
7. T. Xiang and G.A. Gehring, J. Magn. Magn. Mater. 104-107, 861 (1992); T. Xiang and G.A. Gehring (unpublished).
8. See, for example, J.E. Hirsch, Phys. Rev. B 22, 5259 (1980); C. Dasgupta and P. Pfeuty, J. Phys. C 14, 717 (1981), and references therein.

---

<!-- Page 12 -->

9. P.A. Lee, Phys. Rev. Lett. **42**, 1492 (1979).
10. P.A. Lee, and D.S. Fisher, Phys. Rev. Lett. **47**, 882 (1981).
11. R.M. Noack and S.R. White, Phys. Rev. B **47** 9243 (1993).
12. R.P. Feynman, *Statistical Mechanics: A Set of Lectures* (Benjamin, Reading, MA, 1972).
13. W.H. Press, B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling, *Numerical Recipes: The Art of Scientific Computing* (Cambridge University Press, New York, 1986).
14. E.R. Davidson, J. Comput. Phys. **17**, 87 (1975).
15. S.R. White and D. Huse, Phys. Rev. B **48**, 3844 (1993).
16. D. Medeiros and G.G. Cabrera, Phys. Rev. B **43**, 3703 (1991).
17. A. Moreo, Phys. Rev. B **35**, 8562 (1987).
18. S.R. White, Phys. Rev. B **45**, 5752 (1992).
