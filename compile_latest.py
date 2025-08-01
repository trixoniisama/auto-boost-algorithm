import PyInstaller.__main__
from os import listdir
from os.path import dirname, realpath

pwd: str = dirname(realpath(__file__))
files: list[str] = listdir(pwd)
latest: tuple[str, float] = ("", 0.0)
for file in files:
    if file.endswith(".py") and file[-4].isdigit():
        if float(file[-6:-3]) > latest[1]:
            latest = (file, float(file[-6:-3]))

PyInstaller.__main__.run([
    latest[0],
    '--onefile',
    '--hidden-import', 'concurrent',
    '--hidden-import', 'concurrent.futures'
])