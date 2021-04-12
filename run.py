import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def main():
    f = open("requirements.txt", "r")
    packages = f.read().split("\n")
    for package in packages:
        install(package)


if __name__ == "__main__":
    main()
