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

first_card = dbc.Card(
    [
        dbc.CardImg(src="/assets/example.jpeg", top=True),
        dbc.CardBody(
            [
                html.H4("About", className="card-title"),
                html.P(
                    "Person re-identification is the task of associating images of the same " 
                    "person taken from different cameras or from the same camera in different occasions.",
                    className="card-text",
                ),
                dbc.Button("Proof of concept", color="primary", href="https://docs.google.com/presentation/d/1G7WKB74zaWXrSJYQUU6MGq8t2C3h45BOKnVExDfdfuI/edit?usp=sharing"),
            ]
        ),
    ],
    style={
        "width": "20rem",
        "margin-left": "200px",
        "margin-top": "50px"},
)


second_card = dbc.Card(
    [
        dbc.CardImg(src="/assets/dataset_market.jpg", top=True),
        dbc.CardBody(
            [
                html.H4("Dataset", className="card-title"),
                html.P(
                    "The Market-1501 dataset is a dataset for person re-identification, which was collected in" 
                    "front of a supermarket in Tsinghua University. A total of six cameras are used, including 5 high-resolution cameras, and one low-resolution camera. Overlap exists among different cameras." 
                    "Overall, this dataset contains 32,668 annotated bounding boxes of 1,501 identities. Each annotated identity is present in at least two cameras, so that cross-camera search can be performed. ",
                    className="card-text",
                ),
                html.Br(),
                dbc.Button("Dataset Link", color="info", href='https://www.aitribune.com/dataset/2018051063'),
            ]
        ),
    ],
    style={
        "width": "20rem",
        "margin-top": "50px"},
)

third_card = dbc.Card(
    [
        dbc.CardImg(src="/assets/user_guide.jpg", top=True),
        dbc.CardBody(
            [
                html.H4("User manual", className="card-title"),
                html.P(
                    "To inference your image push Upload image button",
                    className="card-text",
                ),
                dbc.Button("Upload image", color="primary", href='/apps/app1'),
            ]
        ),
    ],
    style={
        "width": "20rem",
        "margin-top": "50px"},
)

cards = dbc.Row([dbc.Col(first_card, width="auto"),
                 dbc.Col(second_card, width="auto"),
                 dbc.Col(third_card, width="auto")])

index_page =  html.Div(children=[
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Github Link", href='https://github.com/Lolik-Bolik/Re-Id')),
            dbc.NavItem(dbc.NavLink("Upload image", href='/apps/app1')),
        ],
        brand="Person Re-Identification",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    cards
],
    style={
        "background-image": 'url(/assets/background_2.jpg)',
        "background-repeat": "no-repeat",
        "background-position": "center",
        "background-size": "cover",
        "position": "fixed",
	    "min-height": "100%",
	    "min-width": "100%",})


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