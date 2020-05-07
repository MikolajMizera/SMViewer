# SMViewer
Simple Molecules Viewer is a Python Dash app for displaying sets of molecules 
and filtering them by a property.
## Getting Started
### Dependencies
SMViewer requires:
- dash
- rdkit
- yaml
- numpy
- pandas
### Installing
The main and only required file is `app.py` file. The user needs to specify
`config.yaml` (see examples) and provide SDF file with chemical structures and
properties. The properties should include numerical value to filter by, 
template address to external database, and ID in the database. Please refer to
`app.py` docstring for detailed information.
## Running the examples
For running example on development server, please run app.py script from the 
same directory as the example files.
```
python app.py
```
For production deployment gunicorn is advised (Linux-only).
```
pip install gunicorn
```
```
gunicorn app:server
```
