import nox

@nox.session
def lint(session):
    session.install("flake8")
    session.install("isort")
    session.run("isort", "gaussian.py")
    session.run("isort", "main.py")
    session.run("flake8", "gaussian.py")
    session.run("flake8", "main.py")

