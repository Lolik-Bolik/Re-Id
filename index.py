import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from apps import app1


app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

index_page = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Github Link", href='https://github.com/Lolik-Bolik/Re-Id')),
        dbc.NavItem(dbc.NavLink("Upload image", href='/apps/app1')),
    ],
    brand="Person Re-Identification",
    brand_href="#",
    color="primary",
    dark=True,
)


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    print(pathname)
    if pathname == '/apps/app1':
        return app1.layout
    # elif pathname == '/apps/app2':
    #     return app2.layout
    else:
        return index_page


if __name__ == '__main__':
    app.run_server(debug=True)