import ee
import folium
import geemap

# Initialize the Earth Engine module.
ee.Initialize()

# Define the area of interest (modify coordinates as needed)
aoi = ee.Geometry.Rectangle([-60, -15, -58, -14])

# Load the Hansen Global Forest Change dataset
hansen_dataset = ee.Image('UMD/hansen/global_forest_change_2020_v1_8')

# Select the 'tree cover' layer (percentage of tree cover)
tree_cover = hansen_dataset.select(['treecover2000'])

# Define different tree cover thresholds
thresholds = [30, 50, 90]

# Create forest masks for different thresholds
forest_masks = {t: tree_cover.gt(t) for t in thresholds}

# Create a map
Map = geemap.Map(center=[-14.5, -59], zoom=8)

# Add layers to the map
for t in thresholds:
    mask = forest_masks[t].selfMask()  # Mask non-forest areas
    Map.addLayer(mask, {'palette': '00FF00'}, f'Forest > {t}%')

# Display the map
Map.addLayer(aoi, {}, 'AOI')
Map.addLayerControl()
Map
