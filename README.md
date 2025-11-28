# ECE60131_Project

## TempoRal Intervened Sequences (TRIS) Set-up
- The underlying latent causal process is a dynamic Bayesian network (DBN) $G = (V,E)$ over a set of $K$ causal variables.
  - Each node $i \in V$ is associated with a causal variable $C_i$, which can be scalar or vector valued.
  - Each edge $(i,j)\in E$ represents a causal replation from $C_i$ to $C_j$: $C_i \rightarrow C_j$, where $C_i$ is a parent of $C_j$ and $pa_G (C_i)$ are all parents of $C_i$ in $G.$
- The DBN is first-order Markov, stationary, and without instantaneous effects.
  - We denote the set of all causal variables at time $t$ as $C^{t} = (C^{t}_{1}, \ldots, C^{t}_{K})$, where $C^{t}$ inherits all edges from its components $C_{i}$ for $i \in \llbracket 1..K\rrbracket$ without introducing cycles.
  - In this setting the structure of the graph istime-invariant, i.e., $\mathrm{pa}_{G}(C^{t}_{i}) = \mathrm{pa}_{G}(C^{1}_{i})$ for any $t \in \llbracket 1..T\rrbracket$.
  - For $t \in [1, \dots , T]$ and for each causal factor $i \in [1, \dots , K]$, we can model $C_i = f_i(pa_G (C_i^t), \epsilon_i)$, wehre $pa_G (C_i^t) \include {C_1^{t-1}, \dots, C_k^{t-1}}.$, where all $\epsilon_{i}$ for $i \in \llbracket 1..K\rrbracket$ are mutually independent noises

- We use a binary intervention vector $I^{t} \in \{0,1\}^{K}$ to indicate that a variable $C_{i}^{t}$ in $G$ is intervened upon if and only if $I_{i}^{t} = 1$.
- We consider that the intervention vector components $I_{i}^{t}$ might be confounded by another $I_{j}^{t}$, $i \neq j$, and represent these dependencies with an unobserved regime variable $R^{t}$

- With this, we construct an augmented DAG $G' = (V', E')$, where $
  V' = \{\{C_{i}^{t}\}_{i=1}^{K} \cup \{I_{i}^{t}\}_{i=1}^{K} \cup R^{t}\}_{t=1}^{T}
  $
  and
$
E' = \bigl\{\{\mathrm{pa}_{G}(C_{i}^{t}) \to C_{i}^{t}\}_{i=1}^{K}
      \cup \{I_{i}^{t} \to C_{i}^{t}\}_{i=1}^{K}
      \cup \{R^{t} \to I_{i}^{t}\}_{i=1}^{K}\bigr\}_{t=1}^{T}.
$

- We say that a distribution $p$ is \emph{Markov} w.r.t.\ the augmented DAG $G'$ if it factors as
$
p(V') = \prod_{j \in V'} p\bigl(V_{j} \mid \mathrm{pa}_{G'}(V_{j})\bigr),
$
where $V_{j}$ includes the causal factors $C_{i}^{t}$, the intervention vector components $I_{i}^{t}$, and the regime $R^{t}$. Moreover, we say that $p$ is \emph{faithful} to a causal graph $G'$, if there are no additional conditional independences to the d-separations one can read from the graph $G'$.

- We will consider \emph{soft} interventions, in which the conditional distribution changes, i.e.,
$
p\bigl(C_{i}^{t} \mid \mathrm{pa}_{G}(C_{i}^{t}), I_{i}^{t}=1\bigr)
\neq
p\bigl(C_{i}^{t} \mid \mathrm{pa}_{G}(C_{i}^{t}), I_{i}^{t}=0\bigr),
$
which include as a special case \emph{perfect} interventions $\mathrm{do}(C_{i} = c_{i})$
