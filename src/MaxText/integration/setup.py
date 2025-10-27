from setuptools import setup

setup(name='maxtext_vllm_adapter',
    version='0.1',
    packages=['maxtext_vllm_adapter'],
    entry_points={
        'vllm.general_plugins':
        ["register_maxtext_vllm_adapter = maxtext_vllm_adapter:register"]
    })