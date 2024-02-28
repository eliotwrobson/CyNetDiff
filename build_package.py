import shutil
from pathlib import Path

from setuptools import Extension
from setuptools.dist import Distribution

from Cython.Build import build_ext, cythonize  # isort: skip

# when using setuptools, you should import setuptools before Cython,
# otherwise, both might disagree about the class to use.


def build_cython_extensions():
    # http://docs.cython.org/en/latest/src/userguide/parallelism.html#compiling
    extensions = [
        Extension(
            "cynetdiff.models",
            ["src/cynetdiff/models.pyx"],
            language="c++",
            extra_compile_args=["-O3"],
        ),
    ]

    # include_dirs = set()
    # for extension in extensions:
    #    include_dirs.update(extension.include_dirs)
    # include_dirs = list(include_dirs)

    ext_modules = cythonize(
        extensions,
        annotate=True,
        compiler_directives={"language_level": "3str"},
    )

    dist = Distribution({"ext_modules": ext_modules})
    cmd = build_ext(dist)
    cmd.ensure_finalized()
    cmd.run()

    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = "src" / output.relative_to(cmd.build_lib)
        shutil.copyfile(output, relative_extension)


if __name__ == "__main__":
    build_cython_extensions()
