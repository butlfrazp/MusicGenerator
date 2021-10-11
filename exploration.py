import numpy as np

if __name__ == "__main__":
    arr = np.load("./data/lpd_5/lpd_5_cleansed/A/A/A/TRAAAGR128F425B14B/b97c529ab9ef783a849b896816001748.npz")

    print(arr.files)
    print(arr["info.json"])