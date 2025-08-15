#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import setuptools
from setuptools import find_packages

REQUIREMENTS = [
    'timesfm==1.2.0',
    'torch==2.6.0',
    'jax==0.6.2',
    'google-generativeai==0.8.5',
    'gcsfs==2025.7.0',
    # 'google-cloud-secret-manager==2.24.0',
    # 'google-cloud-bigquery==3.34.0',
]

setuptools.setup(
    name="timesfm-anomaly-pipeline",
    version="1.0",
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    author="Apache Software Foundation",
    author_email="dev@beam.apache.org",
    py_modules=["config"],
)
