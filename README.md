# ECE60131_Project

## TempoRal Intervened Sequences (TRIS) Set-up
- The underlying latent causal process is a dynamic Bayesian network (DBN) $G = (V,E)$ over a set of $K$ causal variables.
  - Each node $i \in V$ is associated with a causal variable $C_i$, which can be scalar or vector valued.
  - Each edge $(i,j)\in E$ represents a causal replation from $C_i$ to $C_j$: $C_i \rightarrow C_j$, where $C_i$ is a parent of $C_j$ and $pa_G (C_i)$ are all parents of $C_i$ in $G.$
- The DBN is first-order Markov, stationary, and without instantaneous effects.
  - We denote the set of all causal variables at time $t$ as $C^t = (C^t_1, \ldots, C^t_K)$, where $C^{t}$ inherits all edges from its components $C_{i}$ for $i \in \[ 1..K\]$ without introducing cycles.
  - In this setting the structure of the graph is time-invariant, i.e., $pa_G(C^t_i) = pa_G(C^1_i)$ for any $t \in \[ 1..T \]$.
  - For $t \in \[ 1, \dots , T\]$ and for each causal factor $i \in \[ 1, \dots , K \]$, we can model $C_i = f_i(pa_G (C_i^t), \epsilon_i)$, wehre $pa_G (C_i^t) \subset {C_1^{t-1}, \dots, C_k^{t-1}}.$, where all $\epsilon_{i}$ for $i \in \[ 1..K \]$ are mutually independent noises

- We use a binary intervention vector $I^{t} \in \{0,1\}^{K}$ to indicate that a variable $C_{i}^{t}$ in $G$ is intervened upon if and only if $I_{i}^{t} = 1$.
- We consider that the intervention vector components $I_{i}^{t}$ might be confounded by another $I_{j}^{t}$, $i \neq j$, and represent these dependencies with an unobserved regime variable $R^{t}$

- With this, we construct an augmented DAG $G' = (V', E')$, where
  ```math
  V' = \{ \{ C_i^t}_{i=1}^K \cup \{ I_i^t\}_{i=1}^K \cup \{ R^t \} \}_{t=1}^T
  ```
  and
  ```math
  E' = \{ \{ pa_G(C_{i}^{t}) \to C_i^t \}_{i=1}^{K}
      \cup \{I_i^t \to C_i^t\}_{i=1}^K
      \cup \{R^{t} \to I_{i}^{t}\}_{i=1}^{K} \}_{t=1}^{T}.
  ```

- We say that a distribution $p$ is Markov w.r.t. the augmented DAG $G'$ if it factors as $p(V') = \prod_{j \in V'} p(V_{j} \mid pa_{G'}(V_{j})),$
where $V_{j}$ includes the causal factors $C_{i}^{t}$, the intervention vector components $I_{i}^{t}$, and the regime $R^{t}$. Moreover, we say that $p$ is faithful to a causal graph $G'$, if there are no additional conditional independences to the d-separations one can read from the graph $G'$.

- We will consider \emph{soft} interventions, in which the conditional distribution changes, i.e.,
  $p(C_{i}^{t} \mid pa_G(C_{i}^{t}), I_{i}^{t}=1) \neq p(C_{i}^{t} \mid pa_G(C_{i}^{t}), I_{i}^{t}=0),$
which include as a special case \emph{perfect} interventions $\mathrm{do}(C_{i} = c_{i})$
