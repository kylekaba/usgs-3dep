import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

def idw_interpolation(coords, z_values, grid_x, grid_y, power=2):
    tree = cKDTree(coords)
    distances, indices = tree.query(np.vstack((grid_x.ravel(), grid_y.ravel())).T, k=4)
    distances = np.maximum(distances, 1e-12)
    weights = 1 / distances**power
    interpolated = np.sum(weights * z_values[indices], axis=1) / np.sum(weights, axis=1)
    return interpolated.reshape(grid_x.shape)

def create_dtm(las_file_path, dtm_output_path, resolution=1.0, power=2):
    # Open the LAS file using laspy
    with laspy.open(las_file_path) as fh:
        las = fh.read()
        # Assuming ground points are classified as 2 in the LiDAR data
        ground_points = las.points[las.classification == 2]
        coords = np.vstack((ground_points.x, ground_points.y)).T
        z_values = ground_points.z
    
    # Create a grid for interpolation
    min_x, max_x, min_y, max_y = las.header.min[0], las.header.max[0], las.header.min[1], las.header.max[1]
    grid_y, grid_x = np.mgrid[max_y:min_y:-resolution,min_x:max_x:resolution]

    # Swap coordinates as well because grid_x and grid_y are swapped
    swapped_Coords = np.vstack((coords[:,1], coords[:,0])).T
    
    # Perform IDW interpolation for Z values on the grid
    dtm_array = idw_interpolation(coords, z_values, grid_x, grid_y, power=power)
    
    # Define the transformation to georeference the raster
    transform = from_origin(west=min_x, north=max_y, xsize=resolution, ysize=-resolution)
    
    # Write the DTM to a GeoTIFF file
    with rasterio.open(
        dtm_output_path, 'w', driver='GTiff',
        height=dtm_array.shape[0], width=dtm_array.shape[1],
        count=1, dtype='float32', crs='EPSG:2227', transform=transform
    ) as dst:
        dst.write(dtm_array, 1)


# Example usage:
# las_file_path = '/Users/kkabasar/Desktop/usgs-3dep/data/USGS_LPC_CA_SantaClaraCounty_2020_A20_09009725.laz'
# dtm_file_path = '/Users/kkabasar/Desktop/usgs-3dep/data/USGS_LPC_CA_SantaClaraCounty_2020_A20_09009725_dtm2.tif'
# create_dtm(las_file_path, dtm_file_path, resolution=1.0, power=2)