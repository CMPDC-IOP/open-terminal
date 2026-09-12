"""Thread defaults shared by every compute-capable child-process launcher."""

import os
from collections.abc import Mapping

# These libraries commonly create a pool sized from the host CPU count unless
# configured before process startup.  Keep the list public so launchers and
# tests use exactly the same contract.
COMPUTE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def with_compute_thread_defaults(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy *environment* and fill missing compute-library thread defaults.

    ``None`` means the current process environment. Existing values, including
    library-specific values, are kept so this remains an overridable hint.
    """
    result = dict(os.environ if environment is None else environment)
    for variable in COMPUTE_THREAD_VARIABLES:
        result.setdefault(variable, "1")
    return result


def compute_thread_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return only the resolved compute variables for a restricted launch."""
    resolved = with_compute_thread_defaults(environment)
    return {variable: resolved[variable] for variable in COMPUTE_THREAD_VARIABLES}
