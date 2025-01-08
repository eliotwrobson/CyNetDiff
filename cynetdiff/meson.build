python.install_sources(
  [
    '__init__.py',
    'utils.py'
  ],
  pure : false,
  subdir : 'cynetdiff'
)

models = python.extension_module(
    'models',
    sources: ['models.pyx'],
    dependencies: [npyrandom_lib, python_dep, numpy_dep],
    override_options : ['cython_language=cpp'],
    install: true,
    subdir: 'cynetdiff',
    #include_directories: [],
)
