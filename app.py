"""
The script with Dash application which displays a set of molecular structures
and allows filtering by a specified value using a slider.

The app is configurable through YAML config files. Configuration parameters
include:

    molecules_sdf: string
    Defines a path to an SDF file with molecular structures to be displayed.

    properties: list of strings
    It contains a list of properties in an SDF which will be used as a
    parameter to filter by. To access view for the selected parameter, the user
    will go to a specific web page. For parameter `pKi` it would be
    `localhost:8050/pKi`.

    db_links: disctionary
    A dictionary with property name as a key and link to database with a
    `%s` placeholder for id.
    Example: for ChEMBL https://www.ebi.ac.uk/chembl/compound_report_card/%s/
    would be used, where `%s` will be replaced by molecule-specific id (see
    below).

    db_id_props: dictionary
    A dictionary with property name as a key and id of corresponding  structure
    in an external database as a value. Will be inserted into the database link
    according to the specified template (see above).

"""

from os.path import join, split

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from base64 import b64encode

import dash
import dash_core_components as dcc
import dash_html_components as html

from rdkit.Chem import Draw
from rdkit import Chem

__author__ = 'Mikołaj Mizera'
__copyright__ = 'Copyright 2020, SMViewer'
__license__ = 'GNU General Public License v3.0'
__version__ = '0.0.1'
__maintainer__ = 'Mikołaj Mizera'
__email__ = 'mikolajmizera@gmail.com'
__status__ = 'PROTOTYPE'


def PrintAsBase64PNGString(mol, legend='', highlightAtoms=[],
                           molSize=(200, 200)):
    """A helper function to depict a moelcular structure and encode it as a
    base64 string."""

    data = Draw._moltoimg(mol,
                          molSize,
                          highlightAtoms,
                          legend,
                          returnPNG=True,
                          kekulize=True)
    return b64encode(data).decode('ascii')


app = dash.Dash(__name__)
server = app.server

# Get execution parmeters from the config.yaml file
config = yaml.safe_load(open('config.yaml'))
mol_file = config['molecules_sdf']
properties = config['properties']
db_links = config['db_links']
db_id_props = config['db_id_props']

# Read all properties from SDF file
mols = [m for m in Chem.SDMolSupplier(mol_file, removeHs=True)]
pbar = tqdm(mols, desc='Loading structures from SDF')
df = pd.DataFrame([m.GetPropsAsDict() for m in pbar]).assign(mol=mols)
df[properties] = df[properties].replace('N/A', np.nan)

# Creates 2D depictions of moelcules and stores them for faster loading
try:
    imgs = np.load('imgs_backup.npy')
except FileNotFoundError:
    pbar = tqdm(df.mol.values, desc='Rendering 2D structures')
    imgs = [PrintAsBase64PNGString(m) for m in pbar]
    np.save('imgs_backup', imgs)
df = df.assign(imgs=imgs)

# Settings of slider will be updated accoridng to values of selected property
slider = dcc.RangeSlider(id='slider',
                         min=0,
                         max=10,
                         step=0.1,
                         value=[0, 10],
                         marks={str(n): {'label': '%.2f'%n}
                                for n in np.arange(11)})

# Layout
header = [
    html.Tr(
            html.Td(
                    html.B('Select pKi range:')
                    )
            ),
    html.Tr(
            html.Td(slider,
                    style={'width': '25%',
                           'padding-right': '10em',
                           'padding-left': '10em'}
                    )
            ),
    html.Tr(
            html.Td([
                html.Td(
                    html.Button('Show',
                                id='button',
                                style={'width': '15%',
                                       'margin-top': '2.5em'}),
                        style={'width': '85%',
                               'margin-top': '2.5em'}),
                html.Td([
                    html.A(html.Img(src=join('assets', 'github_icon.png'),
                                    style={'width':'15px','height':'15px'}),
                           href='https://github.com/MikolajMizera/SMViewer',
                           target='blank'),
                    html.A('SMViewer',
                           href='https://github.com/MikolajMizera/SMViewer',
                           target='blank')],
                    style={'padding-left': '10em',
                           'margin-top': '2.5em',
                           'width': '15%'})
                    ])
            )
    ]


app.layout = html.Div(
    [dcc.Location(id='url', refresh=False),
     html.Table(header, style={'width': '100%'}),
     dcc.Loading(
             html.Table([],
                        id='results',
                        style={'width': '100%',
                               'border-spacing': '1em',
                               'margin-top': '2em',
                               'border-top-width': '1px',
                               'border-top-style': 'dashed',
                               'border-bottom-style': 'dashed',
                               'border-top-color': 'grey'})
                )
    ])

# Updaters - update UI elements accroding to property name
@app.callback(dash.dependencies.Output('slider', 'marks'),
              [dash.dependencies.Input('url', 'pathname')])
def update_slider_marks(pathname):

    if not (pathname is None):
        prop = split(pathname)[-1]
        if prop in properties:
            df_prop = df[prop].dropna().values
            return {('%.2f'%n): {'label': '%.2f'%n}
                    for n in np.linspace(df_prop.min(), df_prop.max(), 10)}

    return {('%.2f'%n): {'label': '%.2f'%n} for n in np.linspace(0, 10, 10)}

@app.callback(dash.dependencies.Output('slider', 'min'),
              [dash.dependencies.Input('url', 'pathname')])
def update_slider_min(pathname):

    if not (pathname is None):
        prop = split(pathname)[-1]
        if prop in properties:
            return df[prop].dropna().min()
    return 0

@app.callback(dash.dependencies.Output('slider', 'max'),
              [dash.dependencies.Input('url', 'pathname')])
def update_slider_max(pathname):

    if not (pathname is None):
        prop = split(pathname)[-1]
        if prop in properties:
            return df[prop].dropna().max()
    return 10

@app.callback(dash.dependencies.Output('slider', 'value'),
              [dash.dependencies.Input('url', 'pathname')])
def update_slider_value(pathname):

    if not (pathname is None):
        prop = split(pathname)[-1]
        if prop in properties:
            return [df[prop].dropna().min(), df[prop].dropna().max()]
    return [0,10]


@app.callback(dash.dependencies.Output('results', 'children'),
              [dash.dependencies.Input('button', 'n_clicks')],
              [dash.dependencies.State('slider', 'value'),
               dash.dependencies.State('url', 'pathname')])
def update_table(n_clicks, prop_range, pathname):

    if (not n_clicks) or (pathname is None):
        return []

    prop = split(pathname)[-1]
    if not (prop in properties):
        return []

    prop_min, prop_max = prop_range

    db_link = db_links[prop]
    db_ids_prop = db_id_props[prop]
    df_prop = df.dropna(subset=[prop])

    mask = (df_prop[prop] >= prop_min) & (df_prop[prop] <= prop_max)
    # limit display to 250 images
    # TODO: add proper configuration parameter
    df_prop = df_prop[mask].sort_values(by=prop).iloc[:250]

    imgs = [html.Img(src='data:image/png;base64,%s'%img)
            for img in df_prop.imgs]

    labels = [html.P(['%s: %.2f'%(prop, pki),
                      html.Br(),
                      html.A('%s'%chid,
                             href=db_link%chid,
                             target='blank')],
                     style={'text-align': 'center'})
              for chid, pki in zip(df_prop[db_ids_prop].values,
                                   df_prop[prop].values)]

    tds = [html.Td([html.Tr(i), html.Tr(l)]) for i, l in zip(imgs, labels)]
    inds = np.array_split(np.arange(len(tds)), int(np.ceil(len(tds)/4)))
    trs = [html.Tr([tds[i] for i in ind]) for ind in inds]

    return trs


if __name__ == '__main__':
    app.run_server(debug=False)
