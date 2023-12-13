#import os
import shutil
from pathlib import Path


# when using setuptools, you should import setuptools before Cython,
# otherwise, both might disagree about the class to use.

from setuptools import Extension  # noqa: I001
from setuptools.dist import Distribution  # noqa: I001
#import Cython.Compiler.Options  # pyright: ignore [reportMissingImports]
from Cython.Build import build_ext, cythonize  # pyright: ignore [reportMissingImports]

#EXTENSIONS = [
#    Extension('leafy.data_structure', ['leafy/data_structure.pyx']),
#    Extension('leafy.graph', ['leafy/graph.pyx']),
#    Extension('leafy.search', ['leafy/search.pyx']),
#    Extension('leafy.digraph', ['leafy/digraph.pyx']),
#    Extension('leafy.shortest_path', ['leafy/shortest_path.pyx']),
#]


def build_cython_extensions():
    extensions = [
        Extension('leafy.data_structure', ['leafy/data_structure.pyx']),
        Extension('leafy.graph', ['leafy/graph.pyx']),
        Extension('leafy.search', ['leafy/search.pyx']),
        Extension('leafy.digraph', ['leafy/digraph.pyx']),
        Extension('leafy.shortest_path', ['leafy/shortest_path.pyx']),
    ]

    #include_dirs = set()
    #for extension in extensions:
    #    include_dirs.update(extension.include_dirs)
    #include_dirs = list(include_dirs)

    ext_modules = cythonize(
        extensions,
        annotate=True,
        compiler_directives={'language_level': "3str"}
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


#TODO need to edit this file to make the build system work right.
