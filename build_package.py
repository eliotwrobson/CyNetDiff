import shutil
from importlib.util import find_spec
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

# TODO follow this example to make building easier:
# https://github.com/FedericoStra/cython-package-example/blob/master/setup.py


def build_cython_extensions() -> None:
    # Flag to enable Cython code generation during install / build. This is
    # enabled during development to generated the C++ files that will be
    # compiled
    USE_CYTHON = find_spec("Cython") is not None

    ext = ".pyx" if USE_CYTHON else ".cpp"

    extensions = [
        Extension(
            "cynetdiff.models",
            ["src/cynetdiff/models" + ext],
            language="c++",
            extra_compile_args=["-O3"],
        ),
    ]

    if USE_CYTHON:
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
