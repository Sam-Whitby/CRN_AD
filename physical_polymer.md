# A physically motivated chain-position model for the dCRN

## The problem with ad-hoc parameters

The current dCRN has three phenomenological parameters whose physical origins are treated independently:

- **φ ∈ [0,1]**: a scalar contact selectivity that uniformly scales all non-native interactions by the same factor
- **s_i ≥ 0**: a per-monomer conformational entropy cost that adds to every contact ΔG involving monomer i
- **ε_‡ ≥ 0**: a global activation barrier that uniformly slows all rate constants

Each captures a real physical effect, but none is derived from any underlying model of polymer structure. They are fit independently, and their relationship to one another or to any measurable molecular property is obscure. φ in particular has no structural interpretation: why should every non-native contact be weaker by exactly the same factor, regardless of which pair of charges is involved?

The deeper problem is that all three parameters try to describe a single underlying physical reality: the geometry of a polymer chain determines which charge groups can contact one another easily, how much conformational entropy is lost when they do, and how rugged the effective energy landscape is that a charge must traverse to escape one partner and find another. A model that derives all three from chain architecture would be both more parsimonious and more predictive.

This report proposes such a model, grounded in polymer physics and reaction-diffusion theory, with a concrete differentiable implementation that is compatible with the existing JAX training infrastructure.

---

## Background: the relevant polymer physics

### Loop closure and conformational entropy (Jacobson–Stockmayer theory)

When two residues at sequence positions $i$ and $j$ form a contact, the chain segment between them must close into a loop. For a Gaussian (freely-jointed) chain of $L = |i - j|$ segments each of Kuhn length $b$, the probability density of the end-to-end vector being at zero (the "J-factor") is:

$$J(L) = \left(\frac{3}{2\pi L b^2}\right)^{3/2}$$

The corresponding conformational entropy cost of loop closure — the free energy penalty for bringing segments $i$ and $j$ into contact starting from an unconstrained random coil — is:

$$\Delta S_{ij} = -k_\mathrm{B} \ln J(L) + \mathrm{const} = \frac{3}{2} k_\mathrm{B} \ln\!\left(\frac{|i-j|}{l_0}\right)$$

for some reference length $l_0$ [Jacobson & Stockmayer 1950; Flory 1969]. This is the theoretical basis for the empirical rule that long-range contacts in proteins and IDPs are entropically penalised relative to local ones.

For a worm-like chain (WLC) with persistence length $l_p$, the expression is modified at short separations ($L \ll l_p$, stiff regime), but the Gaussian limit $\Delta S \sim (3/2) k_\mathrm{B} \ln L$ remains valid whenever $|i-j| \gg l_p/b$, which is satisfied for most IDP contacts of interest ($l_p \approx 1$–$3$ residues for unstructured IDPs). For self-avoiding walks (good solvent), the scaling changes to $\Delta S \sim 2.16\, k_\mathrm{B} \ln L$ in 3D [des Cloizeaux 1980], reflecting Flory exponent $\nu \approx 0.588$.

This immediately gives a physical replacement for the per-monomer entropy parameter: rather than training a free $s_i$ for each monomer, assign each residue a position $x_i \in [0, L_\mathrm{chain}]$ along the chain, and compute the pair entropy from the sequence separation:

$$\Delta S_{ij}^\mathrm{chain} = \frac{3}{2} k_\mathrm{B} \ln\!\left(\frac{|x_i - x_j|}{l_0} + 1\right)$$

(The +1 regularises the divergence as $d \to 0$.) This is differentiable with respect to $x_i$ everywhere.

### First-passage time for intramolecular contact formation

The kinetics of contact formation between residues $i$ and $j$ depend on how long it takes for the chain to bring them within a contact radius $a$, given that they started at a typical end-to-end distance $\sim \sqrt{L} b$.

The classical treatment is due to Szabo, Schulten & Schulten (1980), who solved the first-passage problem exactly for a Gaussian chain in the Smoluchowski (overdamped diffusion) limit. The mean first-passage time is:

$$\langle \tau_{ij} \rangle = \frac{1}{k_\mathrm{SW}} \sim \frac{1}{D a} \cdot J(L)^{-1} \sim \frac{1}{D a} \cdot \left(\frac{Lb^2}{3}\right)^{3/2} \cdot a^{-3}$$

Equivalently, the bimolecular contact rate in the intramolecular (loop-closure) regime is $k_c \sim D a \cdot J(L) = D \cdot a \cdot (3/2\pi L b^2)^{3/2}$.

The key scaling result is:

$$k_\mathrm{contact}(L) \sim L^{-3/2} \quad \text{(Gaussian chain, 3D)}$$

This is the Wilemski–Fixman (1974) result for diffusion-controlled intramolecular reactions. It has been confirmed experimentally for peptide end-to-end contact formation [Hagen & Eaton 1996; Möglich, Joder & Kiefhaber 2006] and for longer disordered chains [Soranno et al. 2012]. The effective rate prefactor for a pair of residues at chain separation $d = |x_i - x_j|$ is therefore:

$$k_0^{ij} = k_0 \cdot \left(\frac{d_0}{d_{ij}}\right)^{3/2}$$

where $d_0$ is a reference separation (e.g., one Kuhn length). This replaces the global $k_0$ with a pair-specific prefactor that is smaller for distant pairs. This is differentiable with respect to $x_i$ (with a soft regularisation near $d = 0$).

### The Rouse/Zimm reconfiguration time

An alternative view comes from the Rouse (1953) model of chain dynamics. The slowest relaxation mode of a chain segment of $L$ monomers has timescale:

$$\tau_R(L) \sim \frac{\zeta b^2 L^{1+2\nu}}{k_\mathrm{B}T}$$

where $\zeta$ is the monomer friction coefficient and $\nu$ is the Flory exponent. For the Zimm model (hydrodynamic interactions included), $\tau_Z \sim L^{3\nu}$. In both cases, distant residue pairs sample the contact distance on a timescale that grows with chain separation as a power law. This reconfiguration timescale sets a natural lower bound on $1/k_0^{ij}$ and provides additional support for the $k_0^{ij} \sim d_{ij}^{-3/2}$ scaling at the level of the mean first-passage time.

---

## The emergent φ: contact selectivity from chain proximity

In the current model, φ is a scalar that uniformly scales all non-native contact energies. Its physical content is: how well does a non-complementary pair of patches fit together geometrically? A value of φ ≈ 0.2 means every wrong-type pair interacts at 20% strength, regardless of which specific pair.

This is physically unsatisfying. In reality, a pair of charges that are close on the chain (say, separated by 5 residues) are much more likely to encounter one another than a pair separated by 100 residues. The effective contact probability is:

$$P_\mathrm{contact}(d_{ij}) \propto J(d_{ij}) \sim d_{ij}^{-3/2}$$

This means that non-native contacts between nearby charges are thermodynamically competitive — they are effectively stronger (more probable) than non-native contacts between distant charges. The effective φ for a given non-native pair (i,j) should therefore depend on the chain separation:

$$\varphi_{ij} = \varphi_0 \cdot f(d_{ij}), \qquad f(d) = \exp\!\left(-\frac{d - d_\mathrm{native}}{\lambda_c}\right)$$

where $\lambda_c$ is a contact locality length scale (of order 10–30 residues for a typical IDP), and $d_\mathrm{native}$ is the separation of the native pair. For native pairs (Ai–Bi), $\varphi_{ij} = 1$ by definition. For non-native pairs, $\varphi_{ij}$ decreases as their chain separation increases, naturally approaching 0 for pairs far apart on the chain.

The important consequence is that **φ is no longer a free parameter**: it is determined by the trained positions $x_i$. The gradient descent optimises the chain arrangement; the contact selectivity profile is a derived property. This eliminates one degree of freedom from the model while grounding the remaining parameters in a structural picture.

---

## Landscape ruggedness from chain topology: eliminating ε_‡

The deepest insight the user identifies is that landscape ruggedness — the slow kinetics attributed to ε_‡ in the current model — should emerge from the effective reaction coordinate for charge-pair formation, which necessarily passes through configurations involving non-specific contacts with competitors.

### The effective reaction coordinate

Consider classifier charge A1 (at position $x_{A1}$) seeking its partner B1 (at position $x_{B1}$). The natural reaction coordinate is not the Euclidean distance between A1 and B1 in 3D space, but rather a projected coordinate along which the chain must reconfigure. As the chain explores configurations, A1 transiently contacts competitor charges a1, a2, b1, b2, … whose positions $x_{a_k}$ are distributed along the chain. Each such transient contact creates a local free-energy minimum that must be escaped before A1 can find B1.

This is precisely the scenario analysed by Zwanzig (1988): diffusion on a rough potential. In Zwanzig's original result, if the barriers are drawn from a Gaussian distribution with variance $\sigma_\varepsilon^2$, the effective diffusion coefficient is:

$$D_\mathrm{eff} = D_0 \cdot \exp\!\left(-\beta^2 \sigma_\varepsilon^2\right)$$

This is valid in any dimension, not just 1D, as argued in the main paper — it is a cumulant-expansion identity. The mean first-passage time for A1 to find B1 is:

$$\langle \tau \rangle \sim \tau_0 \cdot \exp\!\left(\beta^2 \sigma_\varepsilon^2\right)$$

where $\tau_0$ is the roughness-free timescale from the Wilemski–Fixman calculation above.

### The variance from chain architecture

In the chain position model, the variance of non-specific contact energies encountered by A1 on its way to B1 is:

$$\sigma_\varepsilon^2(A1) = \sum_{k \in \mathrm{competitors}} \left(\varphi_{A1,k} \cdot J \cdot q_{A1} \cdot q_k\right)^2 \cdot w\!\left(d_{A1,k}\right)$$

where $w(d) = \exp(-d/\lambda_c)$ is a window function that weights the contribution of competitor $k$ by its proximity to A1 on the chain. This expression captures the key intuition: a competitor that is close on the chain to the native pair contributes strongly to the roughness; one that is far away contributes negligibly.

Crucially, $\sigma_\varepsilon^2$ is differentiable with respect to the positions $x_i$, through both $\varphi_{A1,k} = \varphi_0 \exp(-d_{A1,k}/\lambda_c)$ and $w(d_{A1,k}) = \exp(-d_{A1,k}/\lambda_c)$.

The total kinetic trapping timescale for the native pair A1–B1 is therefore:

$$\tau_\mathrm{trap}^\mathrm{A1-B1} \sim k_0^{-1} \cdot \left(\frac{d_{A1,B1}}{d_0}\right)^{3/2} \cdot \exp\!\left(\beta \left|J \cdot q_{A1} \cdot q_{B1}\right|\right) \cdot \exp\!\left(\beta^2 \sigma_\varepsilon^2(A1)\right)$$

Three terms:
1. **Rouse prefactor** $\sim d_{A1,B1}^{3/2}$: distance-dependent contact rate (replaces $k_0$)
2. **Thermodynamic trapping** $\sim \exp(\beta J |q_{A1} q_{B1}|)$: equilibrium binding depth (controlled by $J$ and pH)
3. **Landscape roughness** $\sim \exp(\beta^2 \sigma_\varepsilon^2)$: Zwanzig correction from competitor density (controlled by chain positions)

ε_‡ is no longer needed: the third term provides the exp($\beta^2 \sigma^2$) kinetic enhancement that ε_‡ was introduced to supply, but now derived from the chain topology rather than assumed as a free parameter.

---

## The connection to IDP charge patterning

This framework connects naturally to an active literature on how the sequence arrangement of charges in IDPs determines their physical properties.

### The κ parameter (Sawle & Ghosh 2015)

Sawle & Ghosh introduced the sequence charge decoration parameter κ ∈ [0,1], which measures how segregated positive and negative charges are along the sequence:

- κ ≈ 0: well-mixed, alternating +/− (maximal short-range contacts between complementary charges)
- κ ≈ 1: fully blocky, +++ … −−− (maximal long-range contacts, globally extended chain)

κ controls the radius of gyration, contact probability distributions, and ultimately phase-separation propensity of IDPs. Zheng et al. (2020) showed via coarse-grained MD that κ has strong effects on IDP dimensions that are predictable from theory.

In the chain position model, optimising $\{x_i\}$ is equivalent to finding an optimal κ-like arrangement for the classification task. The gradient would drive native pairs (Ai, Bi) to be close (local contacts, low entropy penalty, fast association, high effective φ) and competitor pairs to be arranged at intermediate distances — far enough from natives to reduce direct competition, but dense enough near each other to create ruggedness that traps the system in the state established by the correct pH sequence.

### Das & Pappu (2013): the role of opposites

Das & Pappu showed that the segregation of opposite charges in IDPs controls both single-chain compaction and interaction strength with partner molecules. A key result is that the ensemble of accessible conformations — and therefore the range of accessible contact distances — depends sensitively on the charge patterning κ. This is directly relevant: our optimised chain positions determine which contact distances are entropically accessible, which sets both the equilibrium selectivity and the kinetic trapping profile.

### Huihui et al. (2018): asymmetric charge patterning

Huihui and coworkers extended the Sawle–Ghosh analysis to asymmetric polyampholytes (unequal numbers of positive and negative charges), showing that the charge patterning parameter generalises naturally and continues to control chain dimensions and contact probabilities. This is important for the dCRN because in general $N_\mathrm{acid} \neq N_\mathrm{base}$ in the presence of competitor species.

### Contact order and folding (Plaxco, Simons & Baker 1998)

Plaxco et al. showed that the folding rate of two-state proteins correlates strongly with the "relative contact order" (RCO) — the mean chain separation of contacts in the native structure, normalised by chain length:

$$\mathrm{RCO} = \frac{1}{N_c L} \sum_{c \in \mathrm{contacts}} |i_c - j_c|$$

Low contact order (local contacts) → fast folding; high contact order (long-range contacts) → slow folding. This is the kinetic consequence of loop-closure entropy: long-range contacts require more chain reconfiguration and have smaller J-factors. In the dCRN, the gradient descent over $\{x_i\}$ is discovering an effective contact order that optimises classification: native contacts should have low contact order (fast, reliable formation) while the landscape roughness from competitor contacts provides the kinetic memory.

---

## A concrete differentiable implementation

### Trainable parameters

Replace the current parameter set $\{\mathrm{p}K_{a,i}, \varphi, s_i, \varepsilon_\ddagger\}$ with:

$$\theta = \{\mathrm{p}K_{a,i},\ x_i\} \quad i = 1, \ldots, 2N$$

where $x_i \in [0, 1]$ is the normalised position of residue $i$ along a chain of $L_\mathrm{chain}$ residues. Two physical hyperparameters are needed: $\lambda_c$ (contact locality scale, fixed, e.g. 20 residues) and $l_0$ (reference separation for entropy normalisation, fixed, e.g. 1 Kuhn length ≈ 3–4 residues for an IDP).

### Derived quantities

Given positions $\{x_i\}$, all three phenomenological parameters are replaced:

**1. Pair separation**

$$d_{ij} = |x_i - x_j| \cdot L_\mathrm{chain} + \varepsilon_d$$

where $\varepsilon_d \sim 0.5$ residues regularises the singularity at $d = 0$. In JAX:
```python
d_ij = jnp.abs(x[:, None] - x[None, :]) * L_chain + eps_d
```

**2. Contact selectivity matrix (replaces scalar φ)**

For non-native pairs:
$$\varphi_{ij} = \varphi_0 \cdot \exp\!\left(-\frac{d_{ij} - d_\mathrm{ref}}{\lambda_c}\right)$$

For native pairs: $\varphi_{ij} = 1$. The effective ΔG matrix becomes:

$$\Delta G_{ij} = \begin{cases} J \cdot q_i \cdot q_j & (i,j) \in \mathrm{native} \\ \varphi_{ij} \cdot J \cdot q_i \cdot q_j + \Delta S_{ij} & \mathrm{otherwise} \end{cases}$$

**3. Loop-closure entropy (replaces s_i)**

$$\Delta S_{ij} = \frac{3}{2} \cdot \ln\!\left(\frac{d_{ij}}{l_0}\right) \quad \text{[in kT]}$$

This adds an entropy penalty that grows with chain separation, automatically penalising long-range contacts more than short-range ones.

**4. Pair-specific rate prefactor (replaces ε_‡)**

$$k_0^{ij} = k_0 \cdot \left(\frac{d_0}{d_{ij}}\right)^{\alpha}$$

where $\alpha = 3/2$ (Gaussian chain) or $\alpha = 3\nu \approx 1.76$ (self-avoiding chain). The full rates become:

$$k_f^{ij} = k_0^{ij} \cdot \exp\!\bigl(-\beta \max(\Delta G_{ij}, 0)\bigr)$$
$$k_b^{ij} = k_0^{ij} \cdot \exp\!\bigl(-\beta \max(-\Delta G_{ij}, 0)\bigr)$$

Detailed balance is preserved: $k_f^{ij}/k_b^{ij} = \exp(-\beta \Delta G_{ij})$.

**5. Landscape roughness diagnostic (no gradient needed)**

The effective Zwanzig roughness experienced by native pair (A_k, B_k) is:

$$\sigma_k^2 = \sum_{(i,j) \notin \mathrm{native}} \left(\varphi_{ij} J q_i q_j\right)^2 \exp\!\left(-\frac{d_{ij}}{\lambda_c}\right)$$

This quantifies the ruggedness of the landscape around native pair $k$ due to nearby competitors. It is a diagnostic rather than a free parameter — it can be printed after training to characterise the found solution physically.

### What the JAX implementation requires

The only structural change to `crn_ad/physics.py` is that `interaction_energy_matrix()` must accept a **pair-specific φ matrix** $\varphi_{ij}$ (shape $[N, N]$) rather than a scalar φ, and similarly the rate prefactor matrix $k_0^{ij}$ replaces the scalar $k_0$. This is already structurally supported: `jnp.where(correct_mask, V, phi_matrix * V)` instead of `jnp.where(correct_mask, V, phi * V)`.

The entropy contribution $\Delta S_{ij}$ is a matrix added to the full ΔG matrix:
```python
dG = J * qi * qj * phi_matrix + entropy_matrix  # entropy_matrix[i,j] = dS_ij
```

The rate matrices then become:
```python
kf = k0_matrix * jnp.exp(-beta * jnp.maximum(dG, 0.0))
kb = k0_matrix * jnp.exp(-beta * jnp.maximum(-dG, 0.0))
```

All JAX matrix operations, all differentiable.

The `constrain_params()` function gains an `x` parameter:
```python
x = jax.nn.sigmoid(raw['x'])  # positions in [0, 1]
```

and all downstream matrices are computed from $x$ before the existing ODE simulation runs unchanged.

---

## The gradient tells you the optimal Go-model

The key insight is that this model, when trained, discovers an effective Go-like architecture.

In a Go model [Ueda et al. 1975; Bryngelson & Wolynes 1987], only native contacts are stabilised; all others are repulsive. The Go model is the maximally specific energy function for a given native structure — it is the limiting case $\varphi \to 0$ for all non-native pairs. Go models are known to produce fast, reliable folding with minimal kinetic trapping in the wrong state [Clementi et al. 2000; Karanicolas & Brooks 2002], which is precisely the property you do NOT want for a kinetic classifier: you want kinetic trapping in the correct-sequence-dependent state.

What the chain-position gradient descent finds is therefore an **anti-Go** arrangement for the competitors. It places competitors at chain separations that maximise $\sigma_k^2$ — the roughness around the native pairs — so that the system is kinetically trapped after the correct pH sequence, while being free to re-equilibrate after an incorrect sequence. This is the structural basis of the competitor sequestration mechanism, now grounded in polymer geometry rather than an effective scalar φ.

The trained solution will generically have:
1. Native pairs (Ai, Bi) placed close together on the chain (low entropy penalty, fast association, high φ_native → maximally selective)
2. Competitor pairs placed at intermediate distances — not so far that they have no effect (which would reduce roughness), and not so close that they compete directly at equilibrium (which would reduce FDR)
3. The optimal competitor separation is set by the competition between these two effects, discoverable only by gradient descent through the ODE

This is exactly the kind of structured result that makes the model publishable: the trained positions $\{x_i^*\}$ give a structural prediction for where along an IDP sequence the titratable residues should be located. This is a hypothesis that can, in principle, be tested by synthesis or by comparison with natural pH-sensing IDPs.

---

## Connection to AWSEM and coarse-grained IDP models

The model proposed here is related to, but simpler than, the AWSEM (Associative memory, Water-Mediated, Structure and Energy Model) framework [Davtyan et al. 2012; Contessoto et al. 2019]. AWSEM uses a coarse-grained force field with memory terms to describe protein folding; AWSEM-IDP extends this to disordered chains. Both use residue-level position information to compute contact probabilities.

The key difference is that AWSEM is not differentiable with respect to sequence parameters — it is used for forward simulation, not inverse design. The model proposed here strips the full AWSEM force field down to the minimum necessary for differentiable gradient-based training: only the sequence-separation-dependent contact probability and the Boltzmann equilibrium contact energies are retained. This is the appropriate level of coarse-graining for the dCRN, where the goal is to find sequences (charge positions) that optimise classification, not to simulate protein dynamics at atomistic resolution.

A closely related approach is the differentiable coarse-grained model of Janson et al. (2023), which uses Gaussian chain statistics to compute conformational ensembles of IDPs from sequence in a differentiable framework. The present proposal is essentially this approach applied not to structural properties but to kinetic classification — the first such application, to our knowledge.

---

## Limitations and challenges

### 1. One-dimensional chain assumption

Positions $x_i \in [0, L]$ assume the chain is a 1D object. Real IDPs are three-dimensional random coils with a distribution of end-to-end distances. The Gaussian chain model captures the statistics of the distribution correctly (for large enough loops), but the effective 1D projection ignores that two residues can be nearby in 3D while far in sequence, and vice versa. For a first model, 1D positions are appropriate; a 3D extension would require sampling from the conformational ensemble, which breaks differentiability unless approximated by Gaussian averaging (as in Janson et al. 2023).

### 2. Hydrodynamic interactions

The Rouse model ($\tau \sim L^2$) ignores hydrodynamic interactions between chain segments. The Zimm model ($\tau \sim L^{3/2}$ for $\nu = 1/2$, or $\tau \sim L^{3\nu}$ in good solvent) includes these interactions and is generally more accurate for IDPs in solution [Soranno et al. 2012]. The exponent $\alpha$ in $k_0^{ij} \sim d_{ij}^{-\alpha}$ should be treated as a hyperparameter in [3/2, 3ν] ≈ [1.5, 1.76]. The qualitative conclusions are insensitive to this choice.

### 3. Excluded volume

Self-avoiding walk statistics modify the entropy scaling from $(3/2) \ln d$ to $\approx 2.16 \ln d$ in 3D good solvent. For an IDP in physiological salt (moderate screening), the effective Flory exponent is between 0.5 and 0.6. Again, this is a quantitative correction that does not change the qualitative behaviour.

### 4. Position ordering

With $2N$ positions on a 1D chain, the gradient may collapse multiple positions to the same location. A soft repulsion between positions (e.g., $L_\mathrm{repel} = \sum_{i \neq j} \exp(-|x_i - x_j| / \delta)$) added to the loss prevents this and enforces a physically reasonable non-overlapping sequence arrangement.

### 5. The native pair definition

In the current model, native pairs (Ai, Bi) are defined a priori by the mask `correct_mask`. In the chain position model, the concept of a "native pair" needs refinement: if positions are trained, two classifiers that end up very close on the chain might resemble a true "native pair" structurally. One could in principle make the native mask soft (a function of $d_{ij}$), but this changes the loss landscape significantly. A practical approach is to keep the native mask fixed and allow positions to determine only the entropy, φ matrix, and rate prefactor.

---

## Proposed implementation roadmap

### Phase 1: entropy and φ from positions (no rate modification)

1. Add trainable `x_i ∈ [0,1]` for each of $2N$ residues
2. Compute the entropy matrix $\Delta S_{ij}$ and pass it as `monomer_entropy` in matrix form (the current code already supports this via `interaction_energy_matrix`)
3. Compute the φ matrix $\varphi_{ij}$ and replace the scalar `phi` parameter
4. Train and compare FDR with the current model

This is the minimal change. It removes two parameters (scalar φ, per-monomer $s_i$) and replaces them with $2N$ position parameters plus 2 fixed hyperparameters ($\lambda_c$, $l_0$). For $N = 4$, this is 8 new parameters vs 2+4 = 6 old ones — slightly more, but physically grounded.

### Phase 2: pair-specific rate prefactor (Rouse/Wilemski–Fixman)

5. Replace scalar $k_0$ with the matrix $k_0^{ij} = k_0 (d_0/d_{ij})^\alpha$
6. This requires passing $k_0^{ij}$ through `rate_matrices()` — a minor change from scalar to matrix
7. This eliminates ε_‡ and replaces it with position-derived rate heterogeneity

### Phase 3: roughness diagnostic

8. Compute $\sigma_k^2$ for each native pair after training and report it alongside FDR
9. Verify the Zwanzig $\exp(\beta^2 \sigma^2)$ scaling by comparing trained solutions with different $N$, $M$ values

---

## Summary

The ad-hoc parameters φ, $s_i$, and ε_‡ all arise from the same underlying physical reality: the polymer chain architecture of an IDP. Assigning each titratable residue a trainable position $x_i$ along the chain allows all three to be derived simultaneously:

- **Entropy** from Jacobson–Stockmayer loop-closure theory: $\Delta S_{ij} \propto (3/2) \ln |x_i - x_j|$
- **Contact selectivity (φ)** from proximity-dependent encounter probability: $\varphi_{ij} \propto \exp(-|x_i - x_j|/\lambda_c)$
- **Kinetic trapping** from Wilemski–Fixman contact rates: $k_0^{ij} \propto |x_i - x_j|^{-3/2}$, with Zwanzig ruggedness $\sigma_k^2$ emerging from the distribution of competitor positions

The gradient descent finds optimal pKa values AND optimal chain positions simultaneously. The trained positions constitute a structural prediction: a specific arrangement of titratable charges along an IDP sequence that would perform kinetic pH classification. This is directly testable by comparison with natural pH-sensing IDPs, by coarse-grained MD validation, or ultimately by synthesis.

---

## Key references

- Jacobson H, Stockmayer WH (1950). *J. Chem. Phys.* 18, 1600. — Loop closure J-factor, $J(L) \sim L^{-3/2}$
- Wilemski G, Fixman M (1974). *J. Chem. Phys.* 60, 866. — Theory of diffusion-controlled intramolecular reactions
- Szabo A, Schulten K, Schulten Z (1980). *J. Chem. Phys.* 72, 4350. — First passage time for end-to-end contact in Gaussian chains
- Zwanzig R (1988). *Proc. Natl. Acad. Sci.* 85, 2029. — Diffusion in a rough potential; $D_\mathrm{eff} = D_0 \exp(-\beta^2 \sigma^2)$
- Bryngelson JD, Wolynes PG (1987). *Proc. Natl. Acad. Sci.* 84, 7524. — Energy landscape and funnel concept
- Hagen SJ, Eaton WA (1996). *J. Chem. Phys.* 104, 3395. — Peptide end-to-end contact formation rates
- Möglich A, Joder K, Kiefhaber T (2006). *Proc. Natl. Acad. Sci.* 103, 12394. — End-to-end contact formation and chain reconfiguration in IDPs
- Plaxco KW, Simons KT, Baker D (1998). *J. Mol. Biol.* 277, 985. — Contact order and folding rates
- Soranno A et al. (2012). *Proc. Natl. Acad. Sci.* 109, 17800. — Quantitative description of long-range contacts in disordered proteins
- Das RK, Pappu RV (2013). *Proc. Natl. Acad. Sci.* 110, 13392. — Charge patterning and IDP properties
- Sawle L, Ghosh K (2015). *J. Chem. Theory Comput.* 11, 5775. — κ parameter for charge sequence decoration
- Huihui JMJ et al. (2018). *J. Chem. Theory Comput.* 14, 2479. — Asymmetric charge patterning
- Davtyan A et al. (2012). *J. Phys. Chem. B* 116, 8494. — AWSEM coarse-grained force field
- Janson G et al. (2023). — Differentiable coarse-grained model for IDP sequence design
- des Cloizeaux J (1980). *J. Phys. Lett.* 41, L151. — Loop closure entropy for self-avoiding walks, $\Delta S \sim 2.16 k_B \ln L$
- Hyeon C, Thirumalai D (2003). *Proc. Natl. Acad. Sci.* 100, 10249. — Energy landscape roughness in protein folding; $\varepsilon_\ddagger \in [2,5]\,k_\mathrm{B}T$
- Clementi C, Nymeyer H, Onuchic JN (2000). *J. Mol. Biol.* 298, 937. — Go-model protein folding
- Karanicolas J, Brooks CL (2002). *Protein Sci.* 11, 2351. — Go-model folding thermodynamics
- Ueda Y, Taketomi H, Go N (1975). *Int. J. Pept. Protein Res.* 7, 445. — Original Go model
- Rouse PE (1953). *J. Chem. Phys.* 21, 1272. — Rouse chain dynamics
- Zimm BH (1956). *J. Chem. Phys.* 24, 269. — Zimm model with hydrodynamics
- Flory PJ (1969). *Statistical Mechanics of Chain Molecules*. Wiley. — Polymer entropy and Gaussian chain statistics
