from setuptools import setup

package_data = {
    'auglib': [
        'core/bin/*',
    ]
}

setup(
    use_scm_version=True,
    package_data=package_data
)
