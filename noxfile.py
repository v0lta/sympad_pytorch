"""This file contains the configuration for Nox sessions.

This file defines the following Nox sessions:
- test: Installs dependencies, runs setup.py, and runs pytest.
- format: Fixes common convention problems automatically using black and isort.
- lint: Checks code conventions using flake8.
- typing: Checks type hints using mypy.
"""

import nox


@nox.session(name="test")
def test_pad(session):
    session.install("pytest")
    session.install("torch")
    session.install("numpy")
    session.run("python", "setup.py", "install")
    session.run("pytest")


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", ".")
    session.run("black", ".")



@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8")
    session.install(
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.run("flake8", "test", "noxfile.py")

    session.install("sphinx", "doc8")
    session.run("doc8", "--max-line-length", "120", "docs/")


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install("torch")
    session.install("mypy")

    session.run(
        "mypy",
        "--ignore-missing-imports",
        "test")
