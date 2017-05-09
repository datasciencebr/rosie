"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -m rosie` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``rosie.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``rosie.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import click

from rosie.chamber_of_deputies import chamber_of_deputies_main


@click.group()
def run_cli():
    pass


@run_cli.command(help="serenata data path")
@click.option('-t', '--targetpath', type=click.Path(writable=True, dir_okay=True), default="/tmp/serenata-data")
def chamber_of_deputies(targetpath):
    """chamber_of_deputies"""
    chamber_of_deputies_main(targetpath)


main = click.CommandCollection(sources=[run_cli])
