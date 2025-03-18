import subprocess

# Use wget to download the dataset
subprocess.run(["wget", "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])

# Use unzip to extract the dataset
subprocess.run(["unzip", "tiny-imagenet-200.zip", "-d", "data/tiny-imagenet"])
