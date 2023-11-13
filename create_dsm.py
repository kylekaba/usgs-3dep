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

def create_dsm(las_file_path, dsm_output_path, resolution=1.0, power=2):
    # Open the LAS file using laspy
    with laspy.open(las_file_path) as fh:
        las = fh.read()
        coords = np.vstack((las.x, las.y)).T
        z_values = las.z
        
    # Create a grid for interpolation
    min_x, max_x, min_y, max_y = las.header.min[0], las.header.max[0], las.header.min[1], las.header.max[1]
    grid_y, grid_x = np.mgrid[max_y:min_y:-resolution,min_x:max_x:resolution]

    # Swap coordinates as well because grid_x and grid_y are swapped
    swapped_Coords = np.vstack((coords[:,1], coords[:,0])).T
    
    # Perform IDW interpolation for Z values on the grid
    dsm_array = idw_interpolation(coords, z_values, grid_x, grid_y, power=power)
    
    # Define the transformation to georeference the raster
    transform = from_origin(west=min_x, north=max_y, xsize=resolution, ysize=-resolution)

    # Write the DSM to a GeoTIFF file
    with rasterio.open(
        dsm_output_path, 'w', driver='GTiff',
        height=dsm_array.shape[0], width=dsm_array.shape[1],
        count=1, dtype='float32', crs='EPSG:2227', transform=transform
    ) as dst:
        dst.write(dsm_array, 1)


# Example usage
#las_file_path = '/Users/kkabasar/Desktop/usgs-3dep/data/USGS_LPC_CA_SantaClaraCounty_2020_A20_09009725.laz'  # Update this to your LAS file path
#dsm_output_path = '/Users/kkabasar/Desktop/usgs-3dep/data/USGS_LPC_CA_SantaClaraCounty_2020_A20_09009725_dsm2.tif'  # Update this to your desired output path
#create_dsm(las_file_path, dsm_output_path, resolution=1.0)  # Call the function with the desired resolution