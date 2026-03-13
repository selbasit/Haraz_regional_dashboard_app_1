# Hamra Regional Exploration Dashboard

This upgraded Streamlit package is a professional multi-well exploration dashboard built from the uploaded Haraz well files as a working field-scale template for the requested Hamra Field regional exploration dashboard.

## Main Features
- Multitrack well-log panels with synchronized depth
- Lithology overlays and formation shading
- Interactive 3D lithology and wellbore positioning
- Interactive Folium geospatial field map
- Dynamic cross-plots and correlation matrix
- Net pay summary by well and formation
- Formation tops overview tab
- Reservoir quality ranking tab
- Teal-gold professional dashboard theme

## Key Notes
- The uploaded files are Haraz wells, so this bundle uses them as a regional template until the actual Hamra dataset is provided.
- Haraz-2 coordinates are parsed from the LAS header.
- Other well coordinates are relative placeholders for visualization and should be replaced with surveyed field coordinates.
- Formation tops in this demo are screening-style packages derived from depth ranges, not interpreted geological picks.
- Pay flag, porosity, Vsh, Sw, and ranking metrics are quick-look dashboard calculations for exploration screening.

## Run
pip install -r requirements.txt
streamlit run app.py
