# Morphotectonic Analysis

Morphotectonic analysis is a study of the relationship between the Earth's surface features and tectonic processes. This analysis uses geological data, such as fault and ridge traces, to understand the deformation of the Earth's morphology

## Introduction
This code processes geospatial data in Python using GDAL, Rasterio, and Geopandas libraries. It demonstrates different methods of processing DEM data and plotting vector data. The code includes filtering lineaments in a DEM using a directional filter and extracting azimuth from the filtered data. It also includes plotting a windrose for the azimuth data.

## Requirements and Libraries used

Python 3.x

osgeo (gdal) to read the raster data

numpy

matplotlib to visualize the data

rasterio

geopandas to plot the vector data

pandas

windrose for windrose plot


## Code Structure
The code is divided into different sections, each of which has a specific function. The sections are as follows:

1- Importing libraries: This section imports the required libraries for the code.
Setting the working directory: This section sets the working directory to the folder containing the data files.

2- Reading the DEM data: This section reads a digital elevation model (DEM) and converts it to a numpy array. It also replaces any missing values with NaN.

3- Visualizing data with Matplotlib: This section visualizes the DEM data using Matplotlib's 'ColorPlot'.

4- Importing vector data: This section imports vector data (lineaments) in shapefile format using Geopandas. 
It is important to highlight that Lineaments are digitized from already published maps and from scientific papers.
In addition ridges and valleys traces are imported after a geoprocessing with the help of GIS software (skelatonization of the DEM)
It also includes reprojecting the vector data to the same coordinate reference system (CRS) as the DEM.
Opening the raster with Rasterio: This section opens the DEM data using Rasterio and visualizes it with Rasterio's 'show' method. It also adds text labels to the plot to indicate different zones.
Windrose plot for lineaments direction: This section imports azimuth data for lineaments and plots a windrose.
At the final stage, correlation matrix have been generated to analyse the concordance of the different datasets. 
