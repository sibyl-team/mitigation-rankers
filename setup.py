
from setuptools import setup, find_packages

version = 0.2

setup(
    name="mitirankers",
    version=version,
    author="Sibyl Team",
    description="Rankers for epidemic containement",
    url='https://github.com/sibyl-team',
    keywords=["COVID", "COVID-19", "coronavirus", "SARS-CoV-2", "stochastic", "agent-based model", "interventions", "epidemiology",
        "statistical inference"],
    platforms=["OS Independent"],
    packages=find_packages(exclude=["doc","test"]),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        #"sib",
    ]
)
