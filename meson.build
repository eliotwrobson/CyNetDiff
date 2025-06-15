project('cynetdiff', 'cython', 'c',
  meson_version : '>= 1.1',
  default_options : ['warning_level=3']
)

python = import('python').find_installation('python3', pure: false)

if python.language_version().version_compare('< 3.10')
  error('Requires Python >= 3.10')
endif

python_dep = python.dependency()

# https://numpy.org/doc/2.0/reference/random/examples/cython/meson.build.html
_numpy_abs = run_command(python, ['-c',
    'import os; os.chdir(".."); import numpy; print(os.path.abspath(numpy.get_include() + "../../.."))'],
    check: true).stdout().strip()

cc = meson.get_compiler('c')

npymath_path = _numpy_abs / '_core' / 'lib'
npy_include_path = _numpy_abs / '_core' / 'include'
npyrandom_path = _numpy_abs / 'random' / 'lib'
npymath_lib = cc.find_library('npymath', dirs: npymath_path)
npyrandom_lib = cc.find_library('npyrandom', dirs: npyrandom_path)

numpy_dep = dependency('numpy', required: true)

package_name = 'cynetdiff'
# Path to your Python package sources relative to meson.build
package_source_dir = 'src' / package_name

# Install the Cython extension module
python.extension_module(
    'models',
    package_source_dir / 'models.pyx',
    dependencies: [npyrandom_lib, python_dep, numpy_dep],
    override_options : ['cython_language=cpp'],
    install: true,
    subdir: 'cynetdiff',
    cpp_args: ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', '-Wno-pedantic'],
)

# Install pure python files
install_subdir(
  package_source_dir,
  install_dir: python.get_install_dir()
)
