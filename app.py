import os
from os.path import join, split
import sys
print(sys.version)
import numpy as np
import flask
import yaml

from tqdm import tqdm
import dash
print(dash.__file__)
import dash_core_components as dcc
import pandas as pd
import dash_html_components as html
from dash.dependencies import Input, Output
from base64 import b64encode
from rdkit.Chem import Draw
from rdkit import Chem

def PrintAsBase64PNGString(mol, highlightAtoms=[], molSize=(200, 200)):
    data = Draw._moltoimg(mol, molSize, highlightAtoms, '', returnPNG=True,
                          kekulize=True)
    return b64encode(data).decode('ascii')

app = dash.Dash(__name__)
server = app.server

config = yaml.safe_load(open('config.yaml'))
mol_file = config['molecules_sdf']
properties = config['properties']
db_links = config['db_links']
db_id_props = config['db_id_props']

mols = [m for m in Chem.SDMolSupplier(mol_file, removeHs=True)]
df = pd.DataFrame([m.GetPropsAsDict() for m in tqdm(mols)])
df = df.assign(mol=mols)
df[properties] = df[properties].replace('N/A', np.nan)

print('Rendering structures...')
try:
    imgs = np.load('imgs_backup.npy')
except Exception as e:
    print(e)
    imgs = [PrintAsBase64PNGString(m) for m in tqdm(df.mol.values)]
    np.save('imgs_backup', imgs)

df = df.assign(imgs=imgs)

# the slider with initial settings, which will be updated dynamically
slider = dcc.RangeSlider(id='slider',
                         min=0,
                         max=10,
                         step=0.1,
                         value=[0, 10],
                         marks={str(n): {'label': '%.2f'%n}
                                for n in np.arange(11)})

# header contains slider and buttion
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
    [
    dcc.Location(id='url', refresh=False),
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

# Updaters - check if url points to a property and update UI accrodingly
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

@app.callback(
dash.dependencies.Output('slider', 'max'),
[dash.dependencies.Input('url', 'pathname')])
def update_slider_max(pathname):

    if not (pathname is None):
        prop = split(pathname)[-1]
        if prop in properties:
            return df[prop].dropna().max()
    return 10

@app.callback(
dash.dependencies.Output('slider', 'value'),
[dash.dependencies.Input('url', 'pathname')])
def update_slider_value(pathname):

    if not (pathname is None):
        prop = split(pathname)[-1]
        if prop in properties:
            return [df[prop].dropna().min(), df[prop].dropna().max()]
    return [0,10]

@app.callback(
dash.dependencies.Output('results', 'children'),
[dash.dependencies.Input('button', 'n_clicks')],
[dash.dependencies.State('slider', 'value'),
  dash.dependencies.State('url', 'pathname')])
def update_table(n_clicks, prop_range, pathname):

    if (not n_clicks) or (pathname is None):
        return []

    pathname = split(pathname)[-1]
    #check if pathname refers to any known property, otherwise use defaults
    if pathname in properties:
        prop = pathname
    else:
        return []

    prop_min, prop_max = prop_range

    db_link = db_links[prop]
    db_ids_prop = db_id_props[prop]
    df_prop = df.dropna(subset=[prop])

    mask = (df_prop[prop]>=prop_min) & (df_prop[prop]<=prop_max)
    # limit display to 250 images
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