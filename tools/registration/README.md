# Registering the two LiDARs

The real scans combine two LiDAR sensors. Their point clouds were aligned with
point-to-plane ICP from [libpointmatcher](https://github.com/norlab-ulaval/libpointmatcher)
using [icp_config.yaml](icp_config.yaml) (random 50 % subsampling of the
reading cloud, normals from 10 neighbours on the reference cloud, 90 % trimmed
outlier rejection, at most 100 iterations).

```bash
pmicp -v --config icp_config.yaml --isTransfoSaved true \
  --initRotation 0.995724,-0.0923071,-0.0034547,0.0921117,0.995031,-0.0377971,0.0069265,0.0373173,0.999279 \
  --initTranslation -0.0034,-2.43,-0.541 \
  sensor1.vtk sensor2.vtk
```

The initial guess is the approximate pose of sensor 2 relative to sensor 1
(rotation matrix row by row, translation in metres). pmicp writes the refined
transform to `test_complete_transfo.txt`. TODO: describe how this transform
was applied to the sensor 2 frames before merging.
