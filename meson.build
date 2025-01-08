project('cynetdiff', 'cython',
  version : '0.1',
  meson_version : '>= 1.1',
  default_options : ['warning_level=3']
)

pymod = import('python')
python = pymod.find_installation(pure: false)

if python.language_version().version_compare('< 3.9')
  error('Requires Python >= 3.9')
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

subdir('cynetdiff')
subdir('tests')
