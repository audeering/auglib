import os
from setuptools import setup


if os.name == 'nt':
    libname = 'cauglib.dll'
else:
    libname = 'libcauglib.so'

print('auglib/bin/' + libname)
package_data = {
    'auglib': [
        'bin/' + libname,
    ]
}

setup(
    use_scm_version=True,
    package_data=package_data
)
