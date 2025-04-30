# -*- coding: utf-8 -*-
"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
setup.py implementation, interesting because it parsed the first __init__.py and
  extracts the `__author__` and `__version__`
"""

from ast import Assign, Constant, Name, parse
from operator import attrgetter
from os import path
from os.path import extsep

from setuptools import find_packages, setup

package_name = "MaxText"

with open(
  path.join(path.dirname(__file__), f"README{extsep}md"), "rt", encoding="utf8"
) as fh:
  long_description = fh.read()


def main():
  """Main function for setup.py; this actually does the installation"""
  with open(
    path.join(
      path.abspath(path.dirname(__file__)),
      package_name,
      f"__init__{extsep}py",
    ), "rt", encoding="utf8"
  ) as f:
    parsed_init = parse(f.read())

  __author__, __version__, __description__ = map(
    lambda node: node.value if isinstance(node, Constant) else node.s,
    filter(
      lambda node: isinstance(node, Constant),
      map(
        attrgetter("value"),
        filter(
          lambda node: isinstance(node, Assign)
          and any(
            filter(
              lambda name: isinstance(name, Name)
              and name.id
              in frozenset(
                ("__author__", "__version__", "__description__")
              ),
              node.targets,
            )
          ),
          parsed_init.body,
        ),
      ),
    ),
  )

  setup(
    name=package_name,
    author=__author__,
    version=__version__,
    url="https://github.com/AI-Hypercomputer/maxtext",
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
      "Intended Audience :: Developers",
      "License :: OSI Approved :: Apache Software License",
      "Programming Language :: Python :: 3 :: Only",
      "Programming Language :: Python :: 3.10",
      "Programming Language :: Python :: 3.11",
      "Programming Language :: Python :: 3.12",
      "Programming Language :: Python :: 3.13",
      "Programming Language :: ML"
    ],
    license="Apache-2.0",
    license_files=["LICENSE"],
    install_requires=[],
    test_suite=f"{package_name}{path.extsep}tests",
    packages=find_packages()
  )


def setup_py_main():
  """Calls main if `__name__ == '__main__'`"""
  if __name__ == "__main__":
    main()


setup_py_main()
