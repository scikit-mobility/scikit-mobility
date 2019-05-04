# scikit-mobility - Human Mobility analysis in Python
`scikit-mobility` is a library for human mobility analysis in Python. The library allows the user to: 

- represent trajectories, fluxes and spatial tessellations with proper data structures (`TrajDataFrame`, `FlowDataFrame`, `SpatialTessellation`) 
- manage and manipulate mobility data of various formats (call detail records, GPS data, data from Location Based Social Networks, survey data);
- extract human mobility patterns from data, both at individual and collective level (e.g., length of displacements, characteristic distance, origin-destination matrix, etc.)
- generate synthetic individual trajectories using standard mathematical models (random walk models, exploration and preferential return model, etc.)
- generate synthetic fluxes using standard migration models (gravity model, radiation model, etc.)
- assess the privacy risk associated with a mobility dataset

