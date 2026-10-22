# PyAMFF rows

The archive numbers are the paper rows. These templates do not rebuild PyAMFF `8037eb98`. The ranked code is the latest release: package version `0.1.0-beta` at master `240f51d01ca22969369f9bc21760c61cbb3b6665` (4 July 2026, tag `v3.0.0` plus 475 commits).

The twelve cells are EMT Pt38 and C2H4, with Rprop, Adam, and LBFGS, at force coefficient 0 and 0.1. Each cell is one ranked row. NAdam, RAdam, AdamW, and SGD are not archive rows, so they are not on this board.

## Setup

The ranked rows are stock `pyamff` on the archived `config.ini`. The local changes are the ones this release needs in order to start, not a retune: the loss name is `SSE` because `SE` is no longer in the schema, `init_model_parameters` is true because a false flag opens a missing `saved_model_params.pyamff`, and each start gets its own gloo `master_port`. Fingerprints stay on the Fortran engine. The network is 10-7 tanh, NoScaler, one batch, learning rate 0.01, weight decay 0. Weights are the release's unseeded uniform draw on `[-1, 1)`.

Rprop is `torch.optim.Rprop`. Adam is `torch.optim.Adam`. LBFGS is `torch.optim.LBFGS` with `max_iter` 1, `history_size` 10, both solver tolerances `1e-10`, and no line search.

Each archive row keeps its own stop. Force coefficient 0 stops when every parameter-gradient component is below 0.02, or at 10000 epochs. Force coefficient 0.1 stops when every component is below 0.1, or at 30000 epochs. The release also stops when the energy RMSE is below 0.0001 and, if forces are trained, the force RMSE is below 0.01. That RMSE stop is the stock trainer, not an extra threshold.

One harness call is one logged epoch: one full-batch loss and its parameter gradient, including the initial weights. A start that reaches the cap fails. PAR-2 charges that start twice the cap. The legacy `mean_epoch` keeps capped starts at the logged epoch. When the force coefficient is 0, the reported force RMSE is -1, which is how the archive marks an energy row.

Five cells are campaigns already measured on this release. EMT Pt38 Rprop at force coefficient 0 is mean epoch 91.04, 50 of 50, cap 10000. EMT Pt38 LBFGS at force coefficient 0 is stock LBFGS, learning rate 0.01, history 10, no line search, 49 of 50, logged-epoch mean 1748.4, PAR-2 1948.42, tolerance 0.02, cap 10000. EMT Pt38 Adam at force coefficient 0.1 is 50 of 50, mean epoch 1788.26, PAR-2 1788.26, max-abs gradient stop 0.1, cap 30000. C2H4 Rprop at force coefficient 0.1 is mean epoch 11863.06, 16 cap failures, PAR-2 21463.06, cap 30000. EMT Pt38 Rprop at force coefficient 0.1 is the same release and stop, mean epoch 947.86, 50 of 50. The other cells use that same stock command with their own archive stop and cap.

A central difference with step `1e-5` checks the Fortran derivative on the four-atom periodic Pt cell. The largest absolute gap is `1.253631032316349e-09`.
