"""ASV environment that creates the benchmark env with uv.

The uv plugin shipped on ASV's main branch implements ``matches`` as a
classmethod and then reads ``self._uv_path``, which only exists on an
instance, so discovery never accepts it. This module is loaded from
``asv.conf.json`` and uses ``uv venv`` and ``uv pip install``.
"""

from __future__ import annotations

import os
import re
import shutil

from asv import environment, util
from asv.console import log


class Uv(environment.Environment):
    """One virtual environment, built and installed with uv.

    The environment type is ``uv-pip``. ASV's own ``uv`` type creates the
    environment with ``uv venv`` and then installs with ``pip``. Its
    ``matches`` method also reads ``_uv_path`` from the class, where that
    attribute is never set.
    """

    tool_name = "uv-pip"

    def __init__(self, conf, python, requirements, tagged_env_vars):
        self._python = python
        self._requirements = requirements
        super().__init__(conf, python, requirements, tagged_env_vars)
        if shutil.which("uv") is None:
            raise environment.EnvironmentUnavailable("uv command not found")

    @classmethod
    def matches(cls, python):
        if not (re.match(r"^[0-9]", python) or re.match(r"^pypy[0-9.]*$", python)):
            return False
        return shutil.which("uv") is not None

    def _setup(self):
        env = dict(os.environ)
        env.update(self.build_env_vars)
        log.info(f"Creating uv environment {self.name}")
        util.check_call(
            ["uv", "venv", f"--python={self._python}", "--no-project", "--seed", self._path],
            env=env,
        )
        # asv runs install_command by searching the environment's bin, not PATH.
        # The pinned uv is what `uv pip install` in asv.conf.json executes.
        python = os.path.join(self._path, "bin", "python")
        util.check_call(
            ["uv", "pip", "install", "--python", python, "uv==0.13.9"],
            env=env,
        )
        self._install_requirements()

    def _spec(self, key: str, val: str) -> str:
        if key.startswith("pip+"):
            key = key[4:]
        if not val:
            return key
        if val[0] in "=<>!~":
            return f"{key}{val}"
        return f"{key}=={val}"

    def _install_requirements(self):
        specs = [
            self._spec(key, val)
            for key, val in {**self._requirements, **self._base_requirements}.items()
        ]
        if not specs:
            return
        python = os.path.join(self._path, "bin", "python")
        log.info(f"Installing requirements for {self.name}")
        env = dict(os.environ)
        env.update(self.build_env_vars)
        util.check_call(["uv", "pip", "install", "--python", python, *specs], env=env)

    def find_executable(self, executable):
        """Resolve a tool in the env, then on PATH.

        The install command is ``uv pip install``. ``uv`` creates the
        env and is not installed inside it, so the env ``bin`` does not
        contain it.
        """
        try:
            return super().find_executable(executable)
        except OSError:
            found = shutil.which(executable)
            if found:
                return found
            raise

    def run(self, args, **kwargs):
        log.debug(f"Running '{' '.join(args)}' in {self.name}")
        return self.run_executable("python", args, **kwargs)
