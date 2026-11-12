"""Train the latest PyAMFF release and count each loss-gradient evaluation.

One harness call is one full-batch loss and the parameter gradient of that
loss. The network, fingerprints, SSE loss, and optimizer settings are the
stock release. Fingerprints come from the Fortran engine,
``fmodules.fpcalc.calcfps``. A start succeeds when the largest absolute
parameter-gradient component is strictly below the problem tolerance and
the call count is under the cap. A start that reaches the cap fails, and
PAR-2 charges it twice the cap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

SEED_BASE = 20261015
CACHE_ROOT = Path("/tmp/optbench-pyamff/cache")
OPTIMIZERS = ("Adam", "Rprop", "LBFGS")


class _Diverged(RuntimeError):
    """The loss or a parameter gradient left the finite reals."""


def max_abs_converged(abs_max: float, tol: float) -> bool:
    """True when max(abs(gradient)) is strictly below the problem tolerance."""
    return abs_max < tol


def classify_stop(
    *,
    calls: int,
    cap: int,
    converged: bool,
    finite: bool,
    error: str,
) -> tuple[bool, str | None]:
    """Success and the primary failure class for one finished start."""
    if error:
        return False, "crash"
    if not finite:
        return False, "divergence"
    if converged and calls < cap:
        return True, None
    if calls >= cap:
        return False, "cap"
    return False, "wrong_stationary"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_dir(problem_id: str) -> Path:
    return CACHE_ROOT / problem_id.replace("/", "-")


def _fp_paras(root: Path, problem_id: str) -> Path:
    suite, problem = problem_id.split("/", 1)
    path = (
        root
        / "archive"
        / "entries"
        / suite
        / problem
        / "pyamff-adam10-7-fcoeff0"
        / "fpParas.dat"
    )
    if not path.is_file():
        raise FileNotFoundError(f"{problem_id} has no archived fpParas.dat")
    return path


def _section(template: dict) -> dict:
    section = dict(template.get("pyamff") or {})
    if "optimizer" not in section and template.get("algorithm"):
        section["optimizer"] = template["algorithm"]
    if "optimizer" not in section:
        raise ValueError(f"{template['id']} needs pyamff.optimizer")
    name = {"ADAM": "Adam"}.get(str(section["optimizer"]), str(section["optimizer"]))
    if name not in OPTIMIZERS:
        raise ValueError(f"{template['id']} optimizer {name} is outside {OPTIMIZERS}")
    section["optimizer"] = name
    return section


def _stop(problem: dict, template: dict) -> tuple[int, float, float]:
    """Cap, max-abs tolerance, and force coefficient for this template.

    Each archive row carries its own stop. Force coefficient 0 uses the
    energy-only threshold. Force coefficient 0.1 uses the force-training
    threshold. The cap is that row's epoch limit.
    """
    section = _section(template)
    ident = str(template["id"])
    raw_coeff = section.get("force_coefficient")
    if raw_coeff in (None, ""):
        raw_coeff = 0.1 if ident.endswith("-force") or ident.endswith("fcoeff0_1") else 0.0
    coeff = float(raw_coeff)
    stop = template.get("stop") or {}
    if stop.get("max_steps") not in (None, "") and stop.get("gradient_norm") not in (None, ""):
        cap = int(stop["max_steps"])
        tol = float(stop["gradient_norm"])
    elif section.get("epochs_max") not in (None, "") and section.get("loss_grad_tol") not in (None, ""):
        cap = int(section["epochs_max"])
        tol = float(section["loss_grad_tol"])
    elif template.get("failure_cap") not in (None, "") or stop.get("max_steps") not in (None, "") or stop.get("max_force_calls") not in (None, ""):
        raw_cap = template.get("failure_cap")
        if raw_cap in (None, ""):
            raw_cap = stop.get("max_steps", stop.get("max_force_calls"))
        cap = int(raw_cap)
        convergence = problem.get("convergence") or {}
        key = "energy_only_grad_norm" if coeff == 0.0 else "force_training_grad_norm"
        if convergence.get(key) in (None, ""):
            raise ValueError(f"{ident} has a cap and no problem gradient stop")
        tol = float(convergence[key])
    else:
        raise ValueError(f"{ident} needs a stop, a pyamff tolerance, or failure_cap")
    convergence = problem.get("convergence") or {}
    if coeff == 0.0:
        stated_tol = convergence.get("energy_only_grad_norm")
        stated_cap = convergence.get("max_epochs")
    else:
        stated_tol = convergence.get("force_training_grad_norm")
        stated_cap = convergence.get("force_max_epochs")
    if stated_tol not in (None, "") and abs(float(stated_tol) - tol) > 1e-12:
        raise ValueError(f"{ident} gradient_norm is not the problem stop")
    if stated_cap not in (None, "") and int(stated_cap) != cap:
        raise ValueError(f"{ident} max_steps is not the problem epoch cap")
    return cap, tol, coeff


def _full_n(problem: dict) -> int:
    sample = problem.get("sample") or {}
    value = sample.get("initializations")
    if value in (None, "", 0, "0"):
        raise ValueError(f"{problem['id']} has no sample.initializations")
    return int(value)


def prepare(root: Path, problem: dict) -> dict:
    """Fingerprint the problem trajectory once and write one training batch."""
    from optbench.campaign import geometry_archive

    problem_id = str(problem["id"])
    cache = _cache_dir(problem_id)
    cache.mkdir(parents=True, exist_ok=True)
    paras = _fp_paras(root, problem_id)
    traj_src = geometry_archive(root, problem)
    digest = {"fp": _sha256(paras), "traj": _sha256(traj_src)}
    meta_path = cache / "meta.json"
    batch_path = cache / "batches" / "batches_0.pckl"
    if meta_path.is_file() and batch_path.is_file():
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        if stored.get("sha256") == digest:
            return stored

    traj_name = Path(str((problem.get("input") or {}).get("archive_path") or traj_src.name)).name
    if cache.exists():
        for child in cache.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(paras, cache / "fpParas.dat")
    shutil.copy(traj_src, cache / traj_name)
    (cache / "config.ini").write_text(
        "\n".join(
            [
                "[Main]",
                "run_type = trainFF",
                f"Trajectory_file = {traj_name}",
                "use_cohesive_energy = False",
                "energy_training = TRUE",
                "force_training = True",
                "scaler_type = NoScaler",
                "adjust = False",
                "epochs_max = 1",
                "",
                "[Parallelization]",
                "process_num = 1",
                "master_addr = 127.0.0.1",
                "batch_num_per_proc = 1",
                "thread_num = 1",
                "mp_start_method = spawn",
                "",
                "[Fingerprints]",
                "fp_type = BP",
                "fp_engine = Fortran",
                "fp_parameter_file = fpParas.dat",
                "fp_use_existing = False",
                "",
                "[MachineLearningModel]",
                "model_type = neural_network",
                "hidden_layers = 10 7",
                "activation_function = tanh",
                "initial_weights = random_uniform",
                "",
                "[LossFunction]",
                "loss_type = SSE",
                "energy_coefficient = 1.0",
                "force_coefficient = 0.0",
                "",
                "[Optimizer]",
                "optimizer_type = ADAM",
                "learning_rate = 0.01",
                "loss_grad_tol = 0.02",
                "",
                "[CrossValidation]",
                "cross_validation = False",
                "",
                "[Debug]",
                "init_model_parameters = False",
                "",
            ]
        ),
        encoding="utf-8",
    )
    previous = Path.cwd()
    try:
        os.chdir(cache)
        try:
            from pyamff import fmodules
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyAMFF ranked rows use the Fortran fingerprint engine. "
                "Build pyamff.fmodules with f2py and gfortran."
            ) from exc
        if not hasattr(fmodules, "fpcalc"):
            raise RuntimeError("pyamff.fmodules has no fpcalc")
        from pyamff.pyamff_runner import PyAMFFRunner

        runner = PyAMFFRunner(config_file=str(cache / "config.ini"))
        runner.preprocess_data(traj_name)
        fp_range, magnitude_scale, intercept_scale = runner.calc_fingerprints()
        runner.fpRange = fp_range
        runner.set_partitions(fp_range, magnitude_scale, intercept_scale)
    finally:
        os.chdir(previous)
    if not batch_path.is_file():
        raise FileNotFoundError(f"{problem_id} fingerprint prep wrote no batches_0.pckl")
    meta = {
        "problem": problem_id,
        "nimages": int(runner.nImages),
        "nfps": {str(key): int(value) for key, value in runner.nFPs.items()},
        "batch": str(batch_path),
        "sha256": digest,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def _optimizer(name: str, params, lr: float):
    import torch

    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=0.0)
    if name == "Rprop":
        return torch.optim.Rprop(params, lr=lr)
    if name == "LBFGS":
        return torch.optim.LBFGS(
            params,
            lr=lr,
            tolerance_grad=1e-10,
            tolerance_change=1e-10,
            max_iter=1,
            max_eval=None,
            history_size=10,
            line_search_fn=None,
        )
    raise ValueError(f"unknown optimizer {name}")


def _build_model(job: dict, batch):
    import torch
    from pyamff.ml_model_code.loss_functions import SSE
    from pyamff.ml_model_code.pytorch_nn import NeuralNetwork
    from pyamff.utilities.preprocessor import Scaler

    train_forces = float(job["force_coefficient"]) > 0.0
    scaler = Scaler(scalerType="NoScaler", forceTraining=train_forces, loss_type="SE").set_scaler()
    energies = batch.energies.detach().cpu().numpy()
    forces = batch.forces.detach().cpu().numpy()
    natoms = batch.natomsPerImageEnergy.detach().cpu().numpy()
    scaler.scaling(energies, forces, natoms)
    hidden = tuple(int(value) for value in job["hidden"])
    model = NeuralNetwork(
        hiddenlayers=hidden,
        nFPs=dict(job["nfps"]),
        activation=str(job["activation"]),
        forceTraining=train_forces,
        cohE=False,
        params=None,
        scaler=scaler,
        initial_weights="random_uniform",
    )
    model.set_scaler(scaler)
    generator = torch.Generator()
    generator.manual_seed(int(job["seed"]))
    with torch.no_grad():
        for param in model.parameters():
            drawn = torch.rand(param.shape, generator=generator, dtype=torch.float64)
            param.copy_(drawn.mul_(2).sub_(1))
    criterion = SSE(
        loss_type="SE",
        cohe=False,
        energy_coefficient=float(job["energy_coefficient"]),
        force_coefficient=float(job["force_coefficient"]),
        device=torch.device("cpu"),
    )
    return model, criterion


def _grad_stats(grads) -> tuple[float, float, bool]:
    import torch

    l2_max = 0.0
    abs_max = 0.0
    finite = True
    for grad in grads:
        if grad is None or not torch.isfinite(grad).all():
            finite = False
            continue
        l2_max = max(l2_max, float(torch.linalg.vector_norm(grad).item()))
        abs_max = max(abs_max, float(torch.max(torch.abs(grad)).item()))
    return l2_max, abs_max, finite


def _index_planes(batch) -> dict:
    """Cartesian copies of the fingerprint-derivative index, checked once."""
    cached = getattr(batch, "_pyamff_index_plane", None)
    if cached is not None:
        return cached
    import torch

    planes = {}
    for element, index in batch.dEdg_AtomIndices.items():
        plane = index[:, :, 0]
        same = (
            index.ndim == 3
            and index.shape[-1] == 3
            and torch.equal(index[:, :, 1], plane)
            and torch.equal(index[:, :, 2], plane)
        )
        planes[element] = plane if same else None
    batch._pyamff_index_plane = planes
    return planes


def _predict(model, batch, train_forces: bool):
    """Energies and forces for one full batch.

    Force training uses PyAMFF's Behler-Parrinello contraction. The stored
    fingerprint-derivative index repeats the same atom for each Cartesian
    component, so that component is contracted once.
    """
    import torch

    device = torch.device("cpu")
    if not train_forces:
        return model(batch.allElement_fps, batch.dgdx, batch, device)
    nimages = int(batch.nimages)
    energies = torch.zeros((nimages, 1), dtype=torch.float64, device=device)
    pieces = []
    planes = _index_planes(batch)
    for element in model.elements:
        fingerprints = batch.allElement_fps[element].requires_grad_()
        hidden = model.actF(model.nn_models[element]["inputLayer"](fingerprints))
        for name in model.hd_names[element]:
            hidden = model.actF(model.nn_models[element][name](hidden))
        atomic = model.nn_models[element]["outputLayer"](hidden)
        bump = torch.zeros((nimages, 1), dtype=torch.float64, device=device)
        bump.scatter_add_(0, batch.fp_imageIndices[element], atomic)
        energies = torch.add(energies, bump)
        (dedg,) = torch.autograd.grad(
            energies,
            fingerprints,
            grad_outputs=torch.ones_like(energies),
            retain_graph=True,
            create_graph=True,
        )
        flat = dedg.reshape(-1)
        plane = planes[element]
        dgdx = batch.dgdx[element]
        if plane is None:
            pieces.append(torch.sum(torch.mul(dgdx, flat[batch.dEdg_AtomIndices[element]]), 1))
        else:
            pieces.append(torch.einsum("pfc,pf->pc", dgdx, flat[plane]))
    energies = torch.flatten(torch.mul(energies, model.slope))
    total = torch.zeros((int(batch.ntotalAtoms), 3), dtype=torch.float64, device=device)
    total.scatter_add_(0, batch.force_AtomIndices, torch.cat(pieces, 0))
    forces = torch.mul(total, -model.slope)
    return energies, forces


def _evaluate(model, criterion, batch, counter: HarnessCounter, tol: float) -> dict:
    import torch

    counter.charge(())
    params = [param for param in model.parameters()]
    model.zero_grad(set_to_none=True)
    # A zero force coefficient makes the force term identically zero, so the
    # energy loss and its parameter gradient do not use the force graph.
    train_forces = float(criterion.force_coefficient) > 0.0
    pred_energy, pred_forces = _predict(model, batch, train_forces)
    loss = criterion(
        pred_energy,
        pred_forces,
        batch.energies,
        batch.forces,
        n_atoms_energy=batch.natomsPerImageEnergy,
        n_atoms_force=batch.natomsPerImageForce,
        checkF=train_forces,
    )
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    for param, grad in zip(params, grads):
        param.grad = grad
    l2_max, abs_max, finite = _grad_stats(grads)
    energy_loss = float(criterion.energy_loss.detach().item())
    nimages = int(batch.nimages)
    energy_rmse = math.sqrt(energy_loss / nimages) if nimages else math.nan
    force_rmse = None
    if getattr(criterion, "force_loss", None) is not None:
        force_loss = float(criterion.force_loss.detach().item())
        force_rmse = math.sqrt(force_loss / nimages) if nimages else math.nan
    loss_value = float(loss.detach().item())
    if not finite or not math.isfinite(loss_value) or not math.isfinite(energy_rmse):
        raise _Diverged("loss or parameter gradient is not finite")
    return {
        "loss": loss.detach(),
        "loss_value": loss_value,
        "energy_rmse": energy_rmse,
        "force_rmse": force_rmse,
        "l2_max": l2_max,
        "abs_max": abs_max,
        "converged": max_abs_converged(abs_max, tol),
    }


def _train_loaded(job: dict, batch) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    started = time.perf_counter()
    cap = int(job["cap"])
    tol = float(job["tol"])
    counter = HarnessCounter(cache=False, cap=cap)
    error = ""
    finite = True
    converged = False
    energy_rmse = None
    force_rmse = None
    l2_max = None
    abs_max = None
    try:
        model, criterion = _build_model(job, batch)
        params = list(model.parameters())
        optimizer = _optimizer(str(job["optimizer"]), params, float(job["learning_rate"]))
        lbfgs = str(job["optimizer"]) == "LBFGS"

        def closure():
            found = _evaluate(model, criterion, batch, counter, tol)
            closure.state = found
            return found["loss"]

        closure.state = None
        while counter.calls < cap:
            try:
                if lbfgs:
                    optimizer.step(closure)
                    found = closure.state
                else:
                    optimizer.step()
                    found = _evaluate(model, criterion, batch, counter, tol)
            except ForceCallCap:
                break
            except _Diverged:
                finite = False
                break
            if found is None:
                break
            energy_rmse = found["energy_rmse"]
            force_rmse = found["force_rmse"]
            l2_max = found["l2_max"]
            abs_max = found["abs_max"]
            converged = bool(found["converged"])
            if converged or counter.calls >= cap:
                break
            if counter.calls % 1000 == 0:
                print(
                    f"{job['template']} start {job['index']} calls {counter.calls} "
                    f"energy_rmse {energy_rmse:.6g} abs {abs_max:.6g}",
                    flush=True,
                )
    except _Diverged:
        finite = False
    except ForceCallCap:
        pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
    success, failure = classify_stop(
        calls=counter.calls,
        cap=cap,
        converged=converged,
        finite=finite,
        error=error,
    )
    reported_force = -1.0
    if float(job["force_coefficient"]) > 0.0 and force_rmse is not None and math.isfinite(force_rmse):
        reported_force = float(force_rmse)
    record = {
        "schema": "optbench.start/v2",
        "counter": "harness",
        "comparable": True,
        "start": str(job["index"]),
        "index": int(job["index"]),
        "force_calls": int(counter.calls),
        "success": success,
        "failure": failure,
        "template": job["template"],
        "problem": job["problem"],
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
        "algorithm": job["optimizer"],
        "seed": int(job["seed"]),
        "energy_rmse": None if energy_rmse is None or not math.isfinite(energy_rmse) else float(energy_rmse),
        "force_rmse": reported_force,
        "grad_l2_max": None if l2_max is None or not math.isfinite(l2_max) else float(l2_max),
        "grad_abs_max": None if abs_max is None or not math.isfinite(abs_max) else float(abs_max),
        "stop_norm": "max_abs",
        "seconds": time.perf_counter() - started,
    }
    if error:
        record["error"] = error
    path = Path(job["record_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(
        f"{job['template']} start {job['index']} calls {record['force_calls']} "
        f"success {success} failure {failure}",
        flush=True,
    )
    return record


def _run_one(job: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    with Path(job["batch"]).open("rb") as handle:
        batch = pickle.load(handle)
    return _train_loaded(job, batch)


def _job(template: dict, problem: dict, meta: dict, index: int, record_path: Path) -> dict:
    section = _section(template)
    cap, tol, coeff = _stop(problem, template)
    hidden = section.get("hidden_layers") or [10, 7]
    return {
        "index": index,
        "seed": int(section.get("seed_base", SEED_BASE)) + index,
        "batch": meta["batch"],
        "nfps": meta["nfps"],
        "hidden": list(hidden),
        "activation": str(section.get("activation", "tanh")),
        "optimizer": str(section["optimizer"]),
        "learning_rate": float(section.get("learning_rate", 0.01)),
        "energy_coefficient": float(section.get("energy_coefficient", 1.0)),
        "force_coefficient": coeff,
        "cap": cap,
        "tol": tol,
        "template": str(template["id"]),
        "problem": str(problem["id"]),
        "code": str(template.get("code") or "pyamff"),
        "code_version": str(template.get("code_version") or ""),
        "commit": str(template.get("commit") or ""),
        "tuning": str(template.get("tuning") or "default"),
        "record_path": str(record_path),
    }


def _record_dir(output: Path) -> Path:
    return Path("/tmp/optbench-pyamff/records") / output.stem


def _load_done(output: Path) -> dict[int, dict]:
    found: dict[int, dict] = {}
    jsonl = output.with_suffix(".starts.jsonl")
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stop_norm") == "max_abs":
                found[int(row["start"])] = row
    directory = _record_dir(output)
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("stop_norm") == "max_abs":
                found[int(row["start"])] = row
    return found


def _fmt(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _write_dat(path: Path, template: dict, problem: dict, records: list[dict], cap: int, full_n: int) -> dict:
    legacy_counts = [int(row["force_calls"]) if row["success"] else cap for row in records]
    legacy = summarize_capped(legacy_counts, cap)
    metrics = score_v2(
        [
            {
                "force_calls": row["force_calls"],
                "success": row["success"],
                "failure": row["failure"],
            }
            for row in records
        ],
        cap,
    )
    successes = [row for row in records if row["success"]]
    epochs = [float(row["force_calls"]) for row in successes]
    energy = [float(row["energy_rmse"]) for row in successes if row.get("energy_rmse") is not None]
    force = [float(row["force_rmse"]) for row in successes if row.get("force_rmse") is not None]

    def _mean(values: list[float]):
        return sum(values) / len(values) if values else None

    def _median(values: list[float]):
        return float(statistics.median(values)) if values else None

    lines = []
    for key in ("force_calls", "force_calls_min", "force_calls_max"):
        text = _fmt(legacy.get(key))
        if text is not None:
            lines.append(f"{key} {text}")
    if epochs:
        lines.append(f"force_calls_median {_fmt(_median(epochs))}")
    lines.append(f"nfailed {metrics['nfailed']}")
    lines.append(f"starts {len(records)}")
    lines.append(f"code {template.get('code', 'pyamff')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', 'default')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append("counter harness")
    lines.append(f"cap {cap}")
    for key in (
        "par2_mean",
        "median",
        "median_low",
        "median_high",
        "wilson_low",
        "wilson_high",
        "n_cap",
        "n_divergence",
        "n_wrong_stationary",
        "n_crash",
    ):
        text = _fmt(metrics.get(key))
        if text is not None:
            lines.append(f"{key} {text}")
    if epochs:
        lines.append(f"mean_epoch {_fmt(_mean(epochs))}")
        lines.append(f"median_epoch {_fmt(_median(epochs))}")
    if energy:
        lines.append(f"mean_energy_rmse {_fmt(_mean(energy))}")
        lines.append(f"median_energy_rmse {_fmt(_median(energy))}")
    if force:
        lines.append(f"mean_force_rmse {_fmt(_mean(force))}")
        lines.append(f"median_force_rmse {_fmt(_median(force))}")
    if len(records) < full_n:
        lines.append("partial true")
    version = template.get("code_version", "")
    commit = template.get("commit", "")
    lines.append(
        "note "
        f"PyAMFF {version} at {commit}. "
        "Stock optimizer settings from that release. "
        "The stop is max(abs(parameter gradient)) strictly below the problem tolerance. "
        "One harness call is one SSE loss and its parameter gradient. "
        "A capped start scores twice the cap."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return metrics


def _publish(output: Path, template: dict, problem: dict, records: list[dict], cap: int, full_n: int) -> dict:
    ordered = sorted(records, key=lambda row: int(row["start"]))
    metrics = _write_dat(output, template, problem, ordered, cap, full_n)
    public = []
    for row in ordered:
        item = dict(row)
        item.pop("index", None)
        public.append(item)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    provenance = collect_provenance(
        code=str(template.get("code") or "pyamff"),
        code_version=str(template.get("code_version") or ""),
        commit=str(template.get("commit") or ""),
        parameters={
            "optimizer": template.get("algorithm"),
            "pyamff": template.get("pyamff") or {},
            "stop": template.get("stop") or {},
        },
    )
    document = {
        "schema": "optbench.result/v2",
        "problem": problem["id"],
        "template": template["id"],
        "scoring": "par2-wilson-bootstrap",
        "cap": cap,
        "provenance": provenance,
        "metrics": metrics,
        "runs": [{"starts": public, "metrics": metrics}],
    }
    v2_path = output.with_suffix(".v2.json")
    temporary = v2_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(v2_path)
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": metrics.get("success_only_mean"),
        "nfailed": metrics["nfailed"],
        "starts": len(public),
        "par2_mean": metrics.get("par2_mean"),
    }


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Train every initialization and write benchmark.dat beside the starts."""
    del work
    cap, _tol, _coeff = _stop(problem, template)
    full_n = _full_n(problem)
    count = full_n if limit <= 0 else min(full_n, int(limit))
    meta = prepare(root, problem)
    done = _load_done(output) if resume else {}
    directory = _record_dir(output)
    pending = []
    for index in range(count):
        if index in done:
            continue
        pending.append(_job(template, problem, meta, index, directory / f"{index:04d}.json"))
    if pending:
        if workers <= 1:
            fresh = [_run_one(job) for job in pending]
        else:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_run_one, job) for job in pending]
                fresh = []
                total = len(futures)
                for finished, future in enumerate(as_completed(futures), 1):
                    fresh.append(future.result())
                    _publish(output, template, problem, list(done.values()) + fresh, cap, full_n)
                    print(f"{template['id']} {finished}/{total} new starts", flush=True)
        for row in fresh:
            done[int(row["start"])] = row
    records = [done[index] for index in range(count) if index in done]
    if len(records) != count:
        missing = [index for index in range(count) if index not in done]
        raise RuntimeError(f"{template['id']} missing starts {missing[:8]}")
    return _publish(output, template, problem, records, cap, full_n)
