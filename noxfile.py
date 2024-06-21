import nox

@nox.session(name='test')
def test_pad(session):
    session.install('pytest')
    session.install('torch')
    session.install('numpy')
    session.run('python', 'setup.py', 'install')
    session.run('pytest')