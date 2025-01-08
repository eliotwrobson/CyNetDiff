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
numpy_dep = dependency('numpy', required: true)

subdir('cynetdiff')
subdir('tests')
