from setuptools import setup

setup(name='maxtext_vllm_adapter',
    version='0.1',
    packages=['vllm'],
    entry_points={
        'vllm.general_plugins':
        ["maxtext_vllm_adapter = vllm:register"]
    })