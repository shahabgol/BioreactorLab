"""Rendered equations use the exact identifiers exposed by the parameter table."""

import re

from dash import dcc

CORE = r"""
**States and units.** $V$ is L; $N_v,N_d$ are billion cells;
$G,E,L$ are vessel amounts in mmol (glucose, glutamate, lactate).
$g=G/V$, $e=E/V$, $l=L/V$ are mM. $X_v=N_v/V$, $X_t=(N_v+N_d)/V$
are numerically million cells/mL. $[a]_+=\max(0,a)$.

$$ f_g=\frac{g}{\text{k_g}+g},\quad f_e=\frac{e}{\text{k_e}+e},\quad
\rho=\frac{X_t}{\text{capacity}} $$
$$ \mu=\text{mu_max}\,f_g f_e[1-\rho]_+,\qquad
k_d=\text{death}+\text{death_density}\,\rho^2 $$
$$ \frac{dN_v}{dt}=(\mu-k_d)N_v,\quad
\frac{dN_d}{dt}=k_dN_v-\text{lysis}\,N_d $$
$$ \mathrm{VCD}=X_v,\quad\mathrm{TCD}=X_t,\quad
\mathrm{viability}=N_v/(N_v+N_d) $$
$$ q_G=\text{glucose_growth}\,\mu+\text{glucose_maintenance}\,f_g,
\qquad\frac{dG}{dt}=-q_GN_v $$
$$ q_E=\text{glutamate_growth}\,\mu-\text{glutamate_uptake}\,f_e,
\qquad\frac{dE}{dt}=q_EN_v $$
"""
BASELINE = r"""
$$ q_L=\text{lactate_yield}\,q_G-\text{lactate_uptake}
\frac{l}{\text{k_lactate}+l}(1-\mu/\text{mu_max}),\quad
\frac{dL}{dt}=q_LN_v $$

**Fitted:** mu_max, k_g, capacity, death, death_density, glucose_growth,
glucose_maintenance, glutamate_growth, glutamate_uptake, lactate_yield,
lactate_uptake. **Fixed:** k_e, k_lactate, lysis.

**Glutamine and ammonia extension.** $Q,A$ are amounts in mmol and $q=Q/V$
is extracellular glutamine in mM.

$$ f_Q=\frac{q}{\text{k_glutamine}+q},\quad
\frac{dQ}{dt}=(\text{glutamine_growth}\,\mu
-\text{glutamine_uptake}\,f_Q)N_v-\text{glutamine_degradation}\,Q $$
$$ \frac{dA}{dt}=\text{ammonia_growth}\,\mu N_v
+\text{glutamine_degradation}\,Q $$

Fit glutamine_growth, glutamine_uptake and ammonia_growth after freezing the
other kinetics. k_glutamine is fixed at 0.25 mM and glutamine_degradation
at 0 h$^{-1}$ for the fitted baseline. Growth coefficients are mmol/billion
cells and uptake is mmol/(billion cells h). Ammonia uses the provisional NH3
mass basis. Neither species feeds back into growth.
"""
HYBRID = r"""
**Learned species replace the lactate equation:** for each enabled $j$ in
lactate, glutamine and ammonia, $M_j$ is grams and $q_j$ is g/(billion cells h).

$$q_j(z)=\phi_j(z)^T w_j,\quad\frac{dM_j}{dt}=N_vq_j(z),\quad C_j=M_j/V$$

The basis is constant, standardized linear (ridge), or the exported
16-landmark Matern-5/2 GP approximation. Features, coefficients and GP settings
appear in **Features & model**; these are separate from the kinetic-parameter table.

**Fitted core:** the first nine parameters above, through glutamate_uptake.
**Fixed core:** k_e and lysis. lactate_yield, lactate_uptake and k_lactate
are inactive in hybrid mode, even if retained in the parameter export.
Learned species do not feed back into growth.
"""
EVENTS = r"""
**Exact sampling and bolus feeds** (sampling precedes feeds; assays precede both):

$$r=1-v_s/V^-,\quad A^+=rA^-+\sum_fv_fc_{A,f},\quad
V^+=V^--v_s+\sum_fv_f$$

This amount map applies to every solute; cells have zero feed concentrations.
Between events, $dV/dt=0$. RK4 propagates the core on the hourly grid.
The mechanistic nitrogen balances use the core's cell and growth exposure and
positivity-preserving subflows. There is no daily state reset.
Mechanistic metabolites convert to g/L using molecular weights;
learned metabolites retain the assay g/L units. A pre-event assay uses $A^-/V^-$.

Parameter units: mu_max/death/death_density/lysis are h$^{-1}$;
k_g/k_e/k_lactate are mM; capacity is billion cells/L; glucose_growth and
glutamate_growth are mmol/billion cells; maintenance/uptake are
mmol/(billion cells h); lactate_yield is mol/mol.
"""


def equations(hybrid=False):
    content = re.sub(
        r"\$\$(.*?)\$\$",
        lambda m: "\n\n$$\n" + m[1].strip() + "\n$$\n\n",
        CORE + (HYBRID if hybrid else BASELINE) + EVENTS,
        flags=re.S,
    )
    return dcc.Markdown(content, mathjax=True, className="model-equations")
