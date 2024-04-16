import os.path
import pickle
import tarfile
from typing import Any, Callable, Optional, Tuple

import numpy as np
import requests


# Define the URLs and file names
class CIFAR_10():

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    direc = "data/cifar-10-batches-py"
    filename = "cifar-10-python.tar.gz"

    train_data = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    test_data = ["test_batch"]

    # __init__ basics taken from PyTorch cifar.py
    def __init__(
            self,
            root: str,
            #train: bool = True,
            #transform: Optional[Callable] = None,
            #target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:


        if(download):
            self.download(root)

        self.data: Any = []
        self.targets = []

        for i in self.train_data:
            self.unpickle(os.path.join(self.direc,i))
        print(self.data)

    def download(self, root) -> None:
        # Make directory for dataset
        os.makedirs(root, exist_ok=True)
        # Download dataset if we don't have it already
        if not os.path.exists(os.path.join(root,self.filename)):
            print("Downloading CIFAR-10 Dataset...")
            r = requests.get(self.url)
            with open(os.path.join(root,self.filename), "wb") as f:
                f.write(r.content)

        # Extract File
        if not os.path.exists(self.direc):
            print("Extracting CIFAR-10 Dataset...")
            with tarfile.open(os.path.join(root,self.filename), "r:gz") as tar:
                tar.extractall(path=root)
                print("Extract Complete")

    # Baseline Provided by CIFAR-10 Team
    # Unpack the dictionary labels from the training batches
    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
            self.data.append(dict["data"])
            if "labels" in dict:
                self.targets.extend(dict["labels"])
            else:
                self.targets.extend(dict["fine_labels"])
        return dict


CIFAR_10(root='./data', download=True)



