project('cynetdiff', 'cython')

py = import('python').find_installation()
dep_py = py.dependency()

py.extension_module('cynetdiff.models',
                    'src/cynetdiff/models.pyx',
                    dependencies: dep_py)
