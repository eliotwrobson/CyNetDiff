# import os
import shutil
from pathlib import Path

# import Cython.Compiler.Options  # pyright: ignore [reportMissingImports]
from Cython.Build import build_ext, cythonize  # pyright: ignore [reportMissingImports]
from setuptools import Extension  # noqa: I001
from setuptools.dist import Distribution  # noqa: I001

# when using setuptools, you should import setuptools before Cython,
# otherwise, both might disagree about the class to use.


def build_cython_extensions():
    extensions = [
        Extension("ndleafy.data_structure", ["ndleafy/data_structure.pyx"]),
        Extension("ndleafy.graph", ["ndleafy/graph.pyx"]),
        Extension("ndleafy.search", ["ndleafy/search.pyx"]),
        Extension("ndleafy.digraph", ["ndleafy/digraph.pyx"]),
        Extension("ndleafy.shortest_path", ["ndleafy/shortest_path.pyx"]),
    ]

    # include_dirs = set()
    # for extension in extensions:
    #    include_dirs.update(extension.include_dirs)
    # include_dirs = list(include_dirs)

    ext_modules = cythonize(
        extensions, annotate=True, compiler_directives={"language_level": "3str"}
    )

    dist = Distribution({"ext_modules": ext_modules})
    cmd = build_ext(dist)
    cmd.ensure_finalized()
    cmd.run()

    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = output.relative_to(cmd.build_lib)
        shutil.copyfile(output, relative_extension)


if __name__ == "__main__":
    build_cython_extensions()


# TODO need to edit this file to make the build system work right.
