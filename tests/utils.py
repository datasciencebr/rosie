def get_fixtures_dir(file):
    import os

    FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(file)), 'fixtures')
    return FIXTURES_DIR
