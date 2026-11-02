=====
Board
=====


Each row is one code, one version, and one stock method.
The first row on this page is the lowest cost for that problem.
The problem page ranks the same rows inside each method family and on one overall table.
``uv run optbench board`` writes this page from the committed results.
It reads the committed results.

.. raw:: html

   <section class="bench" id="headlines">
   <h2>Lennard-Jones 38 clusters</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="minimization/lj38">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">Newtons Method maxstep=0.25</td><td data-field="family">Minimizers</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">79</td><td data-field="median_calls"></td><td data-field="mean_calls">79</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">1000</td></tr>
   </tbody></table></div>
   <h2>Lennard-Jones 38 clusters with Gaussian force noise</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="minimization/lj38-noise">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">rgmin (0.4.1) · DOGLEG</td><td data-field="family">Minimizers</td><td data-field="family_rank">1</td><td data-field="par2_mean">81.881</td><td data-field="published"></td><td data-field="median_calls">79</td><td data-field="mean_calls">81.881</td><td data-field="hessian_calls">80.881</td><td data-field="fd_calls">9220.43</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">1000</td></tr>
   </tbody></table></div>
   <h2>Morse bulk</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="minimization/morse-bulk">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">rgmin (0.4.1) · NEWTON</td><td data-field="family">Minimizers</td><td data-field="family_rank">1</td><td data-field="par2_mean">8.35</td><td data-field="published"></td><td data-field="median_calls">9</td><td data-field="mean_calls">8.35</td><td data-field="hessian_calls">7.35</td><td data-field="fd_calls"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">100</td></tr>
   </tbody></table></div>
   <h2>Pt heptamer island</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="saddle-search/pt-island">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">Sella (2.6.0) · Sella</td><td data-field="family">Single-ended saddle</td><td data-field="family_rank">1</td><td data-field="par2_mean">5201.62</td><td data-field="published"></td><td data-field="median_calls">225</td><td data-field="mean_calls">247.812</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 4</td><td data-field="n">16</td></tr>
   </tbody></table></div>
   <h2>LJ38 double-ended saddle</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="double-ended-saddle-search/lj38">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Per-image mean</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">OPTIM · DNEB + hybrid eigenvector following</td><td data-field="family">Connect / double-ended</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">131</td><td data-field="median_calls">120</td><td data-field="mean_calls">131</td><td data-field="mean_per_image"></td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">50</td></tr>
   </tbody></table></div>
   <h2>lj38_2ts_separation</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="double-ended-saddle-search/lj38_2ts_separation">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Per-image mean</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">Eon svn (r2549) · Lanczos improved L-BFGS</td><td data-field="family">Minimum mode</td><td data-field="family_rank">1</td><td data-field="par2_mean">502.784</td><td data-field="published"></td><td data-field="median_calls">418</td><td data-field="mean_calls">502.784</td><td data-field="mean_per_image">100.557</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">51</td></tr>
   </tbody></table></div>
   <h2>pt-island</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="double-ended-saddle-search/pt-island">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Per-image mean</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">OPTIM · DNEB + hybrid eigenvector following</td><td data-field="family">Connect / double-ended</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">103</td><td data-field="median_calls">98</td><td data-field="mean_calls">103</td><td data-field="mean_per_image"></td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">59</td></tr>
   </tbody></table></div>
   <h2>Binary Lennard-Jones 100</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj100-binary">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">GMIN · basin-hopping</td><td data-field="family">Basin hopping / global</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">-589.507</td><td data-field="median_calls"></td><td data-field="mean_calls">-589.507</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">100</td></tr>
   </tbody></table></div>
   <h2>lj150</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj150">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">GMIN · core orbits symmetrisation scheme</td><td data-field="family">Basin hopping / global</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">667.38</td><td data-field="median_calls"></td><td data-field="mean_calls">667.38</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">100</td></tr>
   </tbody></table></div>
   <h2>Lennard-Jones 38 global minimum</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj38">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">GMIN (df8da599) · basin-hopping with continuous symmetry measures</td><td data-field="family">Basin hopping / global</td><td data-field="family_rank">1</td><td data-field="par2_mean">5883.83</td><td data-field="published"></td><td data-field="median_calls">3980.5</td><td data-field="mean_calls">5883.83</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">100</td></tr>
   </tbody></table></div>
   <h2>lj75</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj75">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">GMIN · basin-hopping with symmetrise csm moves</td><td data-field="family">Basin hopping / global</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">60692</td><td data-field="median_calls">38118</td><td data-field="mean_calls">60692</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">200</td></tr>
   </tbody></table></div>
   <h2>lj98</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj98">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">GMIN · core orbits symmetrisation scheme</td><td data-field="family">Basin hopping / global</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">1281.92</td><td data-field="median_calls"></td><td data-field="mean_calls">1281.92</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">100</td></tr>
   </tbody></table></div>
   <h2>tip4p10</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/tip4p10">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">GMIN · linear temperature basin-paving</td><td data-field="family">Basin hopping / global</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">16099</td><td data-field="median_calls">11387</td><td data-field="mean_calls">16099</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">10000</td></tr>
   </tbody></table></div>
   <h2>tip4p15</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/tip4p15">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">GMIN · linear temperature basin-paving</td><td data-field="family">Basin hopping / global</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">625362</td><td data-field="median_calls">416929</td><td data-field="mean_calls">625362</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">500</td></tr>
   </tbody></table></div>
   <h2>tip4p20</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/tip4p20">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">GMIN · linear temperature basin-paving</td><td data-field="family">Basin hopping / global</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">2.55667e+07</td><td data-field="median_calls">1.81821e+07</td><td data-field="mean_calls">2.55667e+07</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">100</td></tr>
   </tbody></table></div>
   <h2>lj38</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="lowest-eigenvector/lj38">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">Sella (0.1.1) · lowest eigenvector</td><td data-field="family">Minimum mode</td><td data-field="family_rank">1</td><td data-field="par2_mean">17.8</td><td data-field="published"></td><td data-field="median_calls">17</td><td data-field="mean_calls">17.8</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">200</td></tr>
   </tbody></table></div>
   <h2>LJ38</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="tsopt/LJ38">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">Sella (2.6.0) · Sella</td><td data-field="family">Single-ended saddle</td><td data-field="family_rank">1</td><td data-field="par2_mean">68.74</td><td data-field="published"></td><td data-field="median_calls">59.5</td><td data-field="mean_calls">68.74</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">200</td></tr>
   </tbody></table></div>
   <h2>pt-island</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="tsopt/pt-island">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">Sella (0.1.1) · sella</td><td data-field="family">Single-ended saddle</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">53</td><td data-field="median_calls">51</td><td data-field="mean_calls">53</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">49</td></tr>
   </tbody></table></div>
   <h2>C2H4</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="mlopt/C2H4">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">Pyamff (8037eb98) · pyamff-rprop10-7-fcoeff0</td><td data-field="family">Minimizers</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">184.56</td><td data-field="median_calls"></td><td data-field="mean_calls">184.56</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">0</td></tr>
   </tbody></table></div>
   <h2>EMT Pt 38 nanoparticle</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="mlopt/emt-pt38">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">n2p2 (2.3.0) · Kalman energy</td><td data-field="family">Minimizers</td><td data-field="family_rank">1</td><td data-field="par2_mean">25.02</td><td data-field="published"></td><td data-field="median_calls">14</td><td data-field="mean_calls">25.02</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">50</td></tr>
   </tbody></table></div>
   <h2>Pt-heptamer</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="neb/Pt-heptamer">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Per-image mean</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">eOn (r2090) · eon-lbfgs</td><td data-field="family">Connect / double-ended</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">42</td><td data-field="median_calls"></td><td data-field="mean_calls">42</td><td data-field="mean_per_image"></td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">13</td></tr>
   </tbody></table></div>
   <h2>Al</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="hTST/Al">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">eOn (r2006) · Dimer-CG rotation: max=10 tol=1</td><td data-field="family">Minimum mode</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">21070</td><td data-field="median_calls"></td><td data-field="mean_calls">21070</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">639</td></tr>
   </tbody></table></div>
   <h2>Pt</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="hTST/Pt">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">TSASE · quenchDimer+neb-QM</td><td data-field="family">Band / NEB</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">293200</td><td data-field="median_calls"></td><td data-field="mean_calls">293200</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">13</td></tr>
   </tbody></table></div>
   <h2>Al adatom on Al(100) at 250 K</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="accelerated-md/aladatom">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">eOn · parallel replica</td><td data-field="family">Minimizers</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">2.54294e+08</td><td data-field="median_calls"></td><td data-field="mean_calls">2.54294e+08</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">1</td></tr>
   </tbody></table></div>
   <h2>Al adatom on Al(100) at 150 K</h2>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="accelerated-md/standard_aladatom">
   <thead><tr><th>Row</th><th>Family</th><th>Family rank</th><th>PAR-2</th><th>Published</th><th>Median calls</th><th>Mean calls</th><th>Hessians</th><th>FD calls</th><th>Failures</th><th>Runs</th></tr></thead><tbody>
   <tr><td data-field="label">eOn · parallel replica</td><td data-field="family">Minimizers</td><td data-field="family_rank">1</td><td data-field="par2_mean"></td><td data-field="published">835900</td><td data-field="median_calls"></td><td data-field="mean_calls">835900</td><td data-field="hessian_calls">0</td><td data-field="fd_calls">0</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">1</td></tr>
   </tbody></table></div>
   </section>
   <section id="unparsed"><h2>Not scored</h2><ul><li>results/campaign/eon-3.5.0-de-lj38-ci-neb.dat: no per-start record</li><li>results/campaign/eon-3.5.0-de-lj38-ci-swdneb.dat: no per-start record</li><li>results/campaign/eon-3.5.0-neb-pt-cg.dat: no per-start record</li><li>results/campaign/eon-3.5.0-neb-pt-fire.dat: no per-start record</li><li>results/campaign/eon-3.5.0-neb-pt-lbfgs.dat: no per-start record</li><li>results/campaign/eon-3.5.0-neb-pt-qm.dat: no per-start record</li><li>results/campaign/eon-client-cg.dat: no per-start record</li><li>results/campaign/eon-client-fire.dat: no per-start record</li><li>results/campaign/eon-client-lbfgs-m100.dat: no per-start record</li><li>results/campaign/eon-client-lbfgs-m20.dat: no per-start record</li><li>results/campaign/eon-client-morse-cg.dat: no per-start record</li><li>results/campaign/eon-client-morse-fire.dat: no per-start record</li><li>results/campaign/eon-client-morse-lbfgs.dat: no per-start record</li><li>results/campaign/eon-client-morse-sd.dat: no per-start record</li><li>results/campaign/eon-client-qm.dat: no per-start record</li><li>results/campaign/eon-client-sd-twopoint.dat: no per-start record</li><li>results/campaign/eon-client-sd.dat: no per-start record</li><li>results/campaign/eon-svn-cg.dat: no per-start record</li><li>results/campaign/eon-svn-fire.dat: no per-start record</li><li>results/campaign/eon-svn-lbfgs-m100.dat: no per-start record</li><li>results/campaign/eon-svn-lbfgs-m20.dat: no per-start record</li><li>results/campaign/eon-svn-morse-cg.dat: no per-start record</li><li>results/campaign/eon-svn-morse-fire.dat: no per-start record</li><li>results/campaign/eon-svn-morse-lbfgs.dat: no per-start record</li><li>results/campaign/eon-svn-morse-sd.dat: no per-start record</li><li>results/campaign/eon-svn-qm.dat: no per-start record</li><li>results/campaign/eon-svn-sd-twopoint.dat: no per-start record</li><li>results/campaign/eon-svn-sd.dat: no per-start record</li><li>results/campaign/gmin-paper-lj150-nosym.dat: no per-start record</li><li>results/campaign/gmin-paper-lj150-symmetrise.dat: no per-start record</li><li>results/campaign/gmin-paper-lj38-csm.dat: no per-start record</li><li>results/campaign/gmin-paper-lj38-nosym.dat: no per-start record</li><li>results/campaign/gmin-paper-lj38-symmetrise.dat: no per-start record</li><li>results/campaign/gmin-paper-lj75.dat: no per-start record</li><li>results/campaign/gmin-paper-lj98-nosym.dat: no per-start record</li><li>results/campaign/gmin-paper-lj98-symmetrise.dat: no per-start record</li><li>results/campaign/gmin-paper-tip4p10-tbp.dat: no per-start record</li><li>results/campaign/gmin-paper-tip4p10.dat: no per-start record</li><li>results/campaign/gmin-paper-tip4p15-tbp.dat: no per-start record</li><li>results/campaign/gmin-paper-tip4p15.dat: no per-start record</li><li>results/campaign/gmin-paper-tip4p20-tbp.dat: no per-start record</li><li>results/campaign/gmin-paper-tip4p20.dat: no per-start record</li><li>results/campaign/pele-0.1.2-lbfgs-M1-maxstep0.1.dat: no per-start record</li><li>results/campaign/pele-0.1.2-lbfgs-M10-maxstep0.1.dat: no per-start record</li><li>results/campaign/pele-0.1.2-lbfgs-M100-maxstep0.2.dat: no per-start record</li><li>results/campaign/pele-0.1.2-lbfgs-M20-maxstep0.1.dat: no per-start record</li><li>results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.05.dat: no per-start record</li><li>results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.1.dat: no per-start record</li><li>results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.2.dat: no per-start record</li><li>results/campaign/sella-2.6.0-ts-lj38.dat: no per-start record</li><li>results/asv/ade7c5132e47bb4008b9d4c8ddb26492e9e7287d: AseMorseGpminTuned (&quot;&#x27;0&#x27;&quot;, &quot;&#x27;pos_0000.con&#x27;&quot;) track_force_calls was skipped or failed</li><li>results/asv/1999eab667e0b9e0e4108c2d59c0fb9d9a6fadec: TrackedSearch () track_force_calls was skipped or failed</li><li>results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3: Lj38EonCg (&quot;&#x27;0&#x27;&quot;, &quot;&#x27;lj38-clusters/cluster_0000.con&#x27;&quot;) track_force_calls was skipped or failed</li><li>results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3: Lj38EonFire (&quot;&#x27;0&#x27;&quot;, &quot;&#x27;lj38-clusters/cluster_0000.con&#x27;&quot;) track_force_calls was skipped or failed</li><li>results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3: Lj38EonLbfgsM100 (&quot;&#x27;0&#x27;&quot;, &quot;&#x27;lj38-clusters/cluster_0000.con&#x27;&quot;) track_force_calls was skipped or failed</li><li>results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3: Lj38EonLbfgsM20 (&quot;&#x27;0&#x27;&quot;, &quot;&#x27;lj38-clusters/cluster_0000.con&#x27;&quot;) track_force_calls was skipped or failed</li><li>results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3: Lj38EonQm (&quot;&#x27;0&#x27;&quot;, &quot;&#x27;lj38-clusters/cluster_0000.con&#x27;&quot;) track_force_calls was skipped or failed</li><li>results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3: Lj38EonSd (&quot;&#x27;0&#x27;&quot;, &quot;&#x27;lj38-clusters/cluster_0000.con&#x27;&quot;) track_force_calls was skipped or failed</li><li>results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3: Lj38EonSdTwopoint (&quot;&#x27;0&#x27;&quot;, &quot;&#x27;lj38-clusters/cluster_0000.con&#x27;&quot;) track_force_calls was skipped or failed</li><li>results/asv/d4513f42b39edd0b69ade46ce63fd106e0f406c3: Pending.track_success matched nothing in commit d4513f42b39edd0b69ade46ce63fd106e0f406c3</li></ul></section>
   <section id="unmapped"><h2>Not a problem template</h2><ul><li>quadratic.QuadraticDimer: benchmark quadratic.QuadraticDimer is not a problem template</li><li>quadratic.QuadraticDisplacement: benchmark quadratic.QuadraticDisplacement is not a problem template</li><li>quadratic.QuadraticFailureCap: benchmark quadratic.QuadraticFailureCap is not a problem template</li><li>quadratic.QuadraticFailureCrash: benchmark quadratic.QuadraticFailureCrash is not a problem template</li><li>quadratic.QuadraticFailureDivergence: benchmark quadratic.QuadraticFailureDivergence is not a problem template</li><li>quadratic.QuadraticFailureWrongStationary: benchmark quadratic.QuadraticFailureWrongStationary is not a problem template</li><li>quadratic.QuadraticKept: benchmark quadratic.QuadraticKept is not a problem template</li><li>quadratic.QuadraticNoise0: benchmark quadratic.QuadraticNoise0 is not a problem template</li><li>quadratic.QuadraticNoise1em02: benchmark quadratic.QuadraticNoise1em02 is not a problem template</li><li>quadratic.QuadraticNoise1em03: benchmark quadratic.QuadraticNoise1em03 is not a problem template</li><li>quadratic.QuadraticNoise1em04: benchmark quadratic.QuadraticNoise1em04 is not a problem template</li><li>quadratic.QuadraticProfileCap30: benchmark quadratic.QuadraticProfileCap30 is not a problem template</li><li>quadratic.QuadraticProfileStep001: benchmark quadratic.QuadraticProfileStep001 is not a problem template</li><li>quadratic.QuadraticProfileStep005: benchmark quadratic.QuadraticProfileStep005 is not a problem template</li><li>bench_sella_loweig.Lj38Loweig: benchmark bench_sella_loweig.Lj38Loweig is not a problem template</li></ul></section>
