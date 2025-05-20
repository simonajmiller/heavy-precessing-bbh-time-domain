# heavy-precessing-bbh-time-domain

This repository hosts scripts to download and plot the data from **"Measuring spin precession from massive black holes binaries with gravitational waves: insights from time-domain signal morphology"** by Miller et. al (2025)

The dataset containing posterior samples, waveforms, and other necessary plotting data is available to download from [Zenodo]([url](https://zenodo.org/records/15474960)). 

## Downloading data

After cloning this repository, to download our data release, enter the `data` folder and run the following:
```
chmod +x download_data_from_zenodo.sh
./download_data_from_zenodo.sh
```
This will generate everything you need to run the notebooks in the `figures` and `gifs` folders. The data posteriors in the data release only contain samples for the total mass, mass ratio, effective spin, and effective precessing spin. If you would like posteriors for other parameters, please reach out to smiller@caltech.edu.

Warning: the data release is quite large at 1.8 GB.

## Making figures and animations

In the `figures` folder, run each jupyter notebook to generate the corresponding figure. They are also pre-saved as `.png` files in the directory. 

We have animations corresponding to some figures to present results from more cutoff times and/or simulated signals than are shown in the paper. In the `gifs` folder, run each jupyter notebook to generate these animations. They are also pre-saved as `.gif` files in the directory. 

Note: You need to have the following python packages installed to successfully use the notebooks:
```
pip install h5ify
pip install imageio
pip install seaborn
```
