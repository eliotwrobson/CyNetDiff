python.install_sources(
  [
    '__init__.py',
    'utils.py'
  ],
  pure : false,
  subdir : 'cynetdiff'
)

models = python.extension_module(
    'cynetdiff.models',
    sources: ['models.pyx'],
    dependencies: [numpy_dep, python_dep],
    override_options : ['cython_language=cpp'],
    install: true,
    subdir: 'cynetdiff'
)
