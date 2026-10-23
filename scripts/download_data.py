"""Download and unpack the MovieLens "latest-small" dataset into ./data.

The data is not committed to the repository: GroupLens only allows redistribution
under its own terms, so every user fetches it from the original source.
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
FILES = ("ratings.csv", "movies.csv", "tags.csv")


def download(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if all((target / name).exists() for name in FILES):
        print(f"Dataset already present in {target}")
        return

    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in FILES:
            member = f"ml-latest-small/{name}"
            (target / name).write_bytes(archive.read(member))
    print(f"Saved {', '.join(FILES)} to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("data"))
    download(parser.parse_args().target)
