Aleph capture

streetview/photos/ contains photos; streetview/ holds GeoJSON and plan.svg.
satellite/patches/ contains patches; satellite/ holds GeoJSON and plan.svg.
Open satellite.tif for satellite imagery and map.osm for native map data.
Photo headings are clockwise from north; left/right follow OSM node order.
GeoJSON files record paths, photo locations, or satellite patch footprints.
satellite.tif is a lossless RGBA Cloud Optimized GeoTIFF in EPSG:3857.
It is north-up, cropped to the pixels enclosing the area, with internal overviews.
satellite.png contains the same full-resolution pixels for easy viewing.
Transparent pixels mark missing imagery, unfinished downloads, or pixels outside the area.
terrain.tif holds meters in EPSG:3857 (Int16 up to zoom 12, else Float32).
Keep terrain/tiles/ for resume and offline export.
mesh.glb is Google Earth's textured 3D mesh in meters, resting on y = 0.
Keep mesh/nodes/ for resume and offline export.
Filenames in the manifest and GeoJSON are relative to this run folder.
See manifest.json for settings, sources, imagery dates, and coverage notes.

Credits:
Street View, satellite imagery, and meshes: (c) Google and its providers.
Map data: (c) OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright).
Terrain: AWS Terrain Tiles; credit its sources (https://github.com/tilezen/joerd/blob/master/docs/attribution.md).

Keep this folder together. Continue with: alephgeo capture resume FOLDER
Rebuild images and metadata offline with: alephgeo capture export FOLDER
