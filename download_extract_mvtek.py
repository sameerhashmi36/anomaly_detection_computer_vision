import tarfile
import urllib.request

url = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
dataset_path = 'mvtec_anomaly_detection.tar.xz'
urllib.request.urlretrieve(url, dataset_path)

# Extract the dataset
with tarfile.open(dataset_path, 'r:xz') as file:
    file.extractall('mvtec_anomaly_detection')