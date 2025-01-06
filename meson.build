project('cynetdiff', 'cython')

py = import('python').find_installation()

dep_py = py.dependency()
numpy_dep = dependency('numpy', required: true)

py.extension_module('cynetdiff.models',
                    'src/cynetdiff/models.pyx',
                    dependencies: [numpy_dep, dep_py],
                    override_options : ['cython_language=cpp'],
                    install: true,
                    include_directories: ['src'])
