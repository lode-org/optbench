# Pt-island saddle catalog comparison

Hardware is Intel(R) Xeon(R) Processor, 8 cores, 49342816 kB, cpus 0-7.
Each arm used 1 core and 1 thread. RAYON_NUM_THREADS=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1 and the affinity is one core per method.

179e90f2 dimer-fire reached the goal of 27 connected saddles.
aa6dd937 dimer-fire reached the goal of 27 connected saddles.
179e90f2 dimer-lbfgs reached the goal of 27 connected saddles.
aa6dd937 dimer-lbfgs reached the goal of 27 connected saddles.
179e90f2 dimer-rotskip reached the goal of 27 connected saddles.
aa6dd937 dimer-rotskip reached the goal of 27 connected saddles.
179e90f2 lanczos stopped at 24 connected saddles, below the goal of 27.
aa6dd937 lanczos reached the goal of 27 connected saddles.

| method | 179e90f2 connected/searches | aa6dd937 connected/searches | 179e90f2 force_calls, including the 14 initial-minimisation calls | aa6dd937 force_calls, including the 14 initial-minimisation calls | 179e90f2 calls/search | aa6dd937 calls/search | cap failures 179e90f2 -> aa6dd937 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dimer-fire | 27/13038 | 27/8798 | 13794457 | 9033677 | 1058.0 | 1026.8 | 17->7 |
| dimer-lbfgs | 27/7336 | 27/7336 | 3406018 | 3404696 | 464.3 | 464.1 | 0->0 |
| dimer-rotskip | 27/4763 | 27/4763 | 2128776 | 2129016 | 446.9 | 447.0 | 0->0 |
| lanczos | 24/20000 | 27/18171 | 12598860 | 8445282 | 629.9 | 464.8 | 5->2 |
