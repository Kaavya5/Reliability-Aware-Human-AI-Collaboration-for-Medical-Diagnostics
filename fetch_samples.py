import urllib.request
import os

def download_sample_images():
    """Downloads a few real X-ray samples for easy demoing."""
    data_dir = "data/samples"
    os.makedirs(data_dir, exist_ok=True)
    
    samples = {
        "NORMAL_1.jpeg": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-266.jpg",
        "PNEUMONIA_1.jpeg": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-2.jpg",
        "PNEUMONIA_2.jpeg": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000002-3.jpg",
        "NORMAL_2.jpeg": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000002-268.jpg"
    }
    
    for filename, url in samples.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"Failed to fetch {filename}: {e}")
                
if __name__ == "__main__":
    download_sample_images()
