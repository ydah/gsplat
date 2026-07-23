"""Generate NumPy-authored .npy interoperability fixtures."""

from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "npy"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values = np.arange(6).reshape(2, 3)
    dtypes = {
        "float4": np.dtype("<f4"),
        "float8": np.dtype("<f8"),
        "int4": np.dtype("<i4"),
        "int8": np.dtype("<i8"),
        "b1": np.dtype("|b1"),
    }
    for name, dtype in dtypes.items():
        np.save(OUTPUT_DIR / f"{name}.npy", values.astype(dtype), allow_pickle=False)


if __name__ == "__main__":
    main()
