import os
import platform
import zipfile

import requests
import torch
from tqdm import tqdm


def download():
    base_url = "https://github.com/vboussot/ImpactElastix/releases/download/1.0.0/"
    file = "elastix-impact-{}-shared-with-deps-{}.zip".format(
        "win64" if platform.system() == "Windows" else "linux", "cu126" if torch.cuda.is_available() else "cpu"
    )
    try:
        with requests.get(base_url + file, stream=True, timeout=10) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(file, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {file}",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        print("Download finished.")
        with zipfile.ZipFile(file, "r") as z:
            z.extractall(file.replace(".zip", ""))
        os.remove(file)
    except Exception as e:
        raise e


if __name__ == "__main__":
    download()
