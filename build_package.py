import os
import shutil
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


def build_cython_extensions() -> None:
    # Flag to enable Cython code generation during install / build. This is
    # enabled during development to generated the C++ files that will be
    # compiled
    use_cython = find_spec("Cython") is not None

    ext = ".pyx" if use_cython else ".cpp"

    extensions = [
        Extension(
            "cynetdiff.models",
            ["src/cynetdiff/models" + ext],
            language="c++",
            extra_compile_args=["-O3"],
            include_dirs=[np.get_include()],
            # Not so nice. We need the random/lib library from numpy
            library_dirs=[os.path.join(np.get_include(), "..", "..", "random", "lib")],
            libraries=["npyrandom"],
            define_macros=MACROS,  # type: ignore
        ),
    ]

    if use_cython:
        from Cython.Build import cythonize  # isort: skip

        # when using setuptools, you should import setuptools before Cython,
        # otherwise, both might disagree about the class to use.

        # http://docs.cython.org/en/latest/src/userguide/parallelism.html#compiling

        extensions = cythonize(
            extensions,
            annotate=True,
            compiler_directives={"language_level": "3str"},
        )

    dist = Distribution({"ext_modules": extensions})
    cmd = build_ext(dist)
    cmd.ensure_finalized()
    cmd.run()

    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = "src" / output.relative_to(cmd.build_lib)
        shutil.copyfile(output, relative_extension)


if __name__ == "__main__":
    build_cython_extensions()
