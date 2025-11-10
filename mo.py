import numpy as np
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

class MonteCarloPricing:
    def __init__(self, S0, expiry_days, K, r=0.05, sigma=0.2, simulations=1000, option_type='european', avg_type='arithmetic'):
        self.S0 = float(S0)
        self.K = float(K)
        self.steps = max(10, int(expiry_days))
        self.T = self.steps / 365
        self.r = float(r)
        self.sigma = float(sigma)
        self.simulations = max(100, int(simulations))
        self.option_type = option_type
        self.avg_type = avg_type
        self.dt = self.T / self.steps

    def simulate(self):
        np.random.seed(42)
        steps = self.steps
        sims = self.simulations
        S = np.zeros((steps, sims))
        S[0] = self.S0
        drift = (self.r - 0.5*self.sigma**2)*self.dt
        vol = self.sigma*np.sqrt(self.dt)
        for t in range(1, steps):
            Z = np.random.normal(0, 1, sims)
            S[t] = S[t-1] * np.exp(drift + vol*Z)
        return S

    def calculate_prices(self, S):
        discount = np.exp(-self.r*self.T)
        if self.option_type == 'asian':
            if self.avg_type == 'arithmetic':
                avg_price = np.mean(S, axis=0)
            else:
                avg_price = np.exp(np.mean(np.log(S), axis=0))
            call = discount * np.mean(np.maximum(avg_price - self.K, 0))
            put = discount * np.mean(np.maximum(self.K - avg_price, 0))
        else:
            call = discount * np.mean(np.maximum(S[-1] - self.K, 0))
            put = discount * np.mean(np.maximum(self.K - S[-1], 0))
        return call, put

class MonteCarloEuler:
    def __init__(self, S0, expiry_days, K, r=0.05, sigma=0.2, simulations=10000, steps=365):
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(expiry_days) / 365
        self.r = float(r)
        self.sigma = float(sigma)
        self.simulations = max(100, int(simulations))
        self.steps = max(10, int(steps))
        self.dt = self.T / self.steps

    def simulate_paths(self):
        np.random.seed(42)
        S = np.zeros((self.steps + 1, self.simulations))
        S[0] = self.S0
        for t in range(self.steps):
            Z = np.random.normal(0, 1, self.simulations)
            dW = np.sqrt(self.dt) * Z
            S[t + 1] = S[t] + self.r * S[t] * self.dt + self.sigma * S[t] * dW
        return S

    def price_option(self, S):
        discount = np.exp(-self.r * self.T)
        payoff_call = np.maximum(S[-1] - self.K, 0)
        payoff_put = np.maximum(self.K - S[-1], 0)
        return discount * np.mean(payoff_call), discount * np.mean(payoff_put)
    
class VasicekSimulator:
    def __init__(self, r0, mu, a, sigma, T, steps, sims):
        self.r0 = r0
        self.mu = mu
        self.a = a
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.sims = sims
        self.dt = T / steps

    def simulate_paths(self):
        np.random.seed(42)
        r = np.zeros((self.steps + 1, self.sims))
        r[0] = self.r0

        for t in range(1, self.steps + 1):
            z = np.random.normal(0, 1, self.sims)
            dr = self.a * (self.mu - r[t-1]) * self.dt + self.sigma * np.sqrt(self.dt) * z
            r[t] = r[t-1] + dr

        return r

# Initialisation de l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY], suppress_callback_exceptions=True)
app.title = "Simulateur de Tarification d'Options"

# Styles CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .sidebar {
                position: fixed;
                top: 0;
                left: 0;
                bottom: 0;
                width: 250px;
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                padding: 20px;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
                z-index: 1000;
            }
            .sidebar .nav-link {
                color: #ecf0f1 !important;
                padding: 15px 20px;
                margin: 5px 0;
                border-radius: 8px;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            .sidebar .nav-link:hover {
                background-color: rgba(255,255,255,0.1);
                transform: translateX(5px);
            }
            .sidebar .nav-link.active {
                background-color: #e74c3c;
                color: white !important;
            }
            .main-content {
                margin-left: 250px;
                padding: 30px;
                background-color: #f8f9fa;
                min-height: 100vh;
            }
            .card {
                border: none;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
            }
            .card:hover {
                transform: translateY(-2px);
            }
            .display-4 {
                background: linear-gradient(135deg, #3498db, #2c3e50);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Page d'accueil
home_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📊 Simulateur de Tarification d'Options", className="display-4 fw-bold text-primary mb-4"),
            html.P("Cette application interactive vous permet de simuler et tarifer des options financières en utilisant des modèles stochastiques reconnus en ingénierie financière. Elle est conçue pour les étudiants, chercheurs et professionnels souhaitant explorer la dynamique des prix d'options en fonction de différents paramètres de marché.",
                   className="lead text-dark fs-5 mb-4"),
            html.Hr(className="my-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H3("🧮 Modèles Implémentés", className="text-primary mb-4"),
        ])
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Monte Carlo (Black-Scholes)", className="card-title text-primary"),
                html.P("Simule des trajectoires d'actifs log-normaux pour évaluer des options européennes ou asiatiques avec la méthode de Monte Carlo."),
                html.Ul([
                    html.Li("Options européennes et asiatiques"),
                    html.Li("Moyennes arithmétiques et géométriques"),
                    html.Li("Simulations de mouvements browniens")
                ])
            ])
        ], className="shadow-lg h-100"), md=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Euler-Maruyama (GBM)", className="card-title text-info"),
                html.P("Discrétise les équations différentielles stochastiques pour simuler l'évolution de prix d'actifs avec la méthode d'Euler-Maruyama."),
                html.Ul([
                    html.Li("Discrétisation temporelle"),
                    html.Li("Schéma d'Euler explicite"),
                    html.Li("Simulations de trajectoires de prix")
                ])
            ])
        ], className="shadow-lg h-100"), md=4),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Vasicek (Modèle de taux)", className="card-title text-success"),
                html.P("Génère des scénarios stochastiques de taux d'intérêt à moyenne réversion pour l'analyse obligataire et la gestion du risque de taux."),
                html.Ul([
                    html.Li("Modèle de taux à moyenne réversion"),
                    html.Li("Processus d'Ornstein-Uhlenbeck"),
                    html.Li("Simulations de trajectoires de taux")
                ])
            ])
        ], className="shadow-lg h-100"), md=4),
    ], className="gy-4 mb-5"),

    dbc.Row([
        dbc.Col([
            html.H3("🎯 Mode d'emploi", className="text-primary mb-3"),
            html.Ul([
                html.Li("Sélectionnez un modèle via le menu de navigation latéral."),
                html.Li("Saisissez les paramètres de simulation (prix actuel, strike, échéance, etc.)."),
                html.Li("Ajustez les paramètres de volatilité et de taux selon vos besoins."),
                html.Li("Lancez la simulation et analysez les résultats générés dynamiquement."),
                html.Li("Visualisez les graphiques interactifs et les prix calculés.")
            ], className="fs-5 text-dark")
        ])
    ]),

    html.Hr(className="my-5"),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("💡 À propos des modèles", className="text-primary"),
                html.P("Tous les modèles utilisent une génération de nombres aléatoires avec une graine fixe pour assurer la reproductibilité des résultats. Les calculs sont optimisés pour fournir des résultats rapides et précis."),
                html.P("Les paramètres par défaut sont configurés pour des simulations réalistes, mais vous pouvez les ajuster librement pour explorer différents scénarios de marché.")
            ], className="p-4 bg-light rounded-3")
        ])
    ])
], fluid=True)

# Layout Monte Carlo
monte_carlo_layout = dbc.Container([
    html.H2("📊 Simulations Monte Carlo - Modèle Black-Scholes", className="text-primary my-4"),
    
    dbc.Card([
        dbc.CardBody([
            html.H4("Paramètres de Simulation", className="card-title text-primary mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Prix actuel de l'actif (S₀) €", html_for="opt-S0"),
                    dbc.Input(id="opt-S0", value="100", type="number", min="0.01", step="0.01"),
                    dbc.FormText("Prix actuel de l'actif sous-jacent")
                ], md=3),
                dbc.Col([
                    dbc.Label("Prix d'exercice (K) €", html_for="opt-strike"),
                    dbc.Input(id="opt-strike", value="105", type="number", min="0.01", step="0.01"),
                    dbc.FormText("Strike price de l'option")
                ], md=3),
                dbc.Col([
                    dbc.Label("Jours jusqu'à échéance", html_for="opt-expiry-days"),
                    dbc.Input(id="opt-expiry-days", value="30", type="number", min="1", step="1"),
                    dbc.FormText("Durée en jours jusqu'à l'échéance")
                ], md=3),
                dbc.Col([
                    dbc.Label("Type d'option", html_for="opt-type"),
                    dcc.Dropdown(
                        id='opt-type',
                        options=[
                            {'label': 'Européenne', 'value': 'european'}, 
                            {'label': 'Asiatique', 'value': 'asian'}
                        ], 
                        value='european'
                    )
                ], md=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Taux sans risque (%)", html_for="opt-rate"),
                    dbc.Input(id="opt-rate", value="5.0", type="number", min="0", step="0.1"),
                    dbc.FormText("Taux d'intérêt annuel sans risque")
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatilité (%)", html_for="opt-vol"),
                    dbc.Input(id="opt-vol", value="20.0", type="number", min="0.1", step="0.1"),
                    dbc.FormText("Volatilité annuelle de l'actif")
                ], md=3),
                dbc.Col([
                    dbc.Label("Nombre de simulations", html_for="opt-sims"),
                    dbc.Input(id="opt-sims", value="1000", type="number", min="100", step="100"),
                    dbc.FormText("Nombre de trajectoires à simuler")
                ], md=3),
                dbc.Col([
                    dbc.Label("Type de moyenne", html_for="opt-avg-type"),
                    dcc.Dropdown(
                        id='opt-avg-type',
                        options=[
                            {'label': 'Arithmétique', 'value': 'arithmetic'},
                            {'label': 'Géométrique', 'value': 'geometric'}
                        ], 
                        value='arithmetic',
                        disabled=False
                    ),
                    dbc.FormText("Uniquement pour options asiatiques")
                ], md=3)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "🚀 Lancer la Simulation", 
                        id="opt-calculate", 
                        color="primary", 
                        size="lg",
                        className="w-100 mt-3"
                    )
                ], md=12)
            ])
        ])
    ], className="shadow-lg mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📈 Graphique des Simulations", className="mb-0")),
                dbc.CardBody(dcc.Graph(id="opt-paths"))
            ], className="shadow-lg")
        ], md=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("💰 Résultats", className="mb-0 text-success")),
                dbc.CardBody([
                    html.Div(id="opt-results", className="fs-5")
                ])
            ], className="shadow-lg h-100")
        ], md=4)
    ], className="mt-4")
], fluid=True)

# Layout Euler-Maruyama
euler_layout = dbc.Container([
    html.H2("📉 Simulation Euler-Maruyama - Modèle Black-Scholes", className="text-primary my-4"),
    
    dbc.Card([
        dbc.CardBody([
            html.H4("Paramètres de Simulation", className="card-title text-primary mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Prix actuel de l'actif (S₀) €", html_for="euler-S0"),
                    dbc.Input(id="euler-S0", value="100", type="number", min="0.01", step="0.01")
                ], md=3),
                dbc.Col([
                    dbc.Label("Prix d'exercice (K) €", html_for="euler-strike"),
                    dbc.Input(id="euler-strike", value="105", type="number", min="0.01", step="0.01")
                ], md=3),
                dbc.Col([
                    dbc.Label("Jours jusqu'à échéance", html_for="euler-expiry-days"),
                    dbc.Input(id="euler-expiry-days", value="30", type="number", min="1", step="1")
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatilité (%)", html_for="euler-vol"),
                    dbc.Input(id="euler-vol", value="20.0", type="number", min="0.1", step="0.1")
                ], md=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Taux sans risque (%)", html_for="euler-rate"),
                    dbc.Input(id="euler-rate", value="5.0", type="number", min="0", step="0.1")
                ], md=3),
                dbc.Col([
                    dbc.Label("Nombre de simulations", html_for="euler-sims"),
                    dbc.Input(id="euler-sims", value="5000", type="number", min="100", step="100")
                ], md=3),
                dbc.Col([
                    dbc.Label("Pas de temps", html_for="euler-steps"),
                    dbc.Input(id="euler-steps", value="365", type="number", min="10", step="1"),
                    dbc.FormText("Nombre de pas par simulation")
                ], md=3),
                dbc.Col([
                    dbc.Button(
                        "🚀 Lancer la Simulation", 
                        id="euler-calc", 
                        color="primary", 
                        size="lg",
                        className="w-100 mt-4"
                    )
                ], md=3)
            ])
        ])
    ], className="shadow-lg mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📈 Trajectoires Simulées", className="mb-0")),
                dbc.CardBody(dcc.Graph(id="euler-paths"))
            ], className="shadow-lg")
        ], md=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("💰 Prix des Options", className="mb-0 text-success")),
                dbc.CardBody([
                    html.Div(id="euler-prices", className="fs-5")
                ])
            ], className="shadow-lg h-100")
        ], md=4)
    ])
], fluid=True)

# Layout Vasicek
vasicek_layout = dbc.Container([
    html.H2("📈 Modèle de Vasicek - Simulation des Taux d'Intérêt", className="text-primary my-4"),
    
    dbc.Card([
        dbc.CardBody([
            html.H4("Paramètres du Modèle de Vasicek", className="card-title text-primary mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Taux initial (r₀) %", html_for="vasicek-r0"),
                    dbc.Input(id="vasicek-r0", type="number", value="3.0", step="0.1"),
                    dbc.FormText("Taux d'intérêt initial")
                ], md=3),
                dbc.Col([
                    dbc.Label("Moyenne long terme (μ) %", html_for="vasicek-mu"),
                    dbc.Input(id="vasicek-mu", type="number", value="3.0", step="0.1"),
                    dbc.FormText("Niveau moyen de long terme")
                ], md=3),
                dbc.Col([
                    dbc.Label("Vitesse de rappel (a)", html_for="vasicek-a"),
                    dbc.Input(id="vasicek-a", type="number", value="0.5", step="0.01", min="0.01"),
                    dbc.FormText("Vitesse de retour à la moyenne")
                ], md=3),
                dbc.Col([
                    dbc.Label("Volatilité (σ) %", html_for="vasicek-sigma"),
                    dbc.Input(id="vasicek-sigma", type="number", value="1.5", step="0.1", min="0.1"),
                    dbc.FormText("Volatilité des taux")
                ], md=3)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Maturité (T, années)", html_for="vasicek-T"),
                    dbc.Input(id="vasicek-T", type="number", value="2.0", step="0.1", min="0.1")
                ], md=3),
                dbc.Col([
                    dbc.Label("Nombre de simulations", html_for="vasicek-sims"),
                    dbc.Input(id="vasicek-sims", type="number", value="1000", min="100", step="100")
                ], md=3),
                dbc.Col([
                    dbc.Label("Pas de temps", html_for="vasicek-steps"),
                    dbc.Input(id="vasicek-steps", type="number", value="365", min="10", step="1"),
                    dbc.FormText("Pas de discrétisation temporelle")
                ], md=3),
                dbc.Col([
                    dbc.Button(
                        "🚀 Lancer la Simulation", 
                        id="vasicek-simulate", 
                        color="primary", 
                        size="lg",
                        className="w-100 mt-4"
                    )
                ], md=3)
            ])
        ])
    ], className="shadow-lg mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Trajectoires des Taux d'Intérêt", className="mb-0")),
                dbc.CardBody(dcc.Graph(id="vasicek-graph"))
            ], className="shadow-lg")
        ], md=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("ℹ️ Informations de Simulation", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="vasicek-info", className="fs-5 text-success")
                ])
            ], className="shadow-lg mt-4")
        ], md=12)
    ])
], fluid=True)

# Structure globale avec navigation
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    
    html.Div([
        html.Div([
            html.H2("📈 Stochastic Explorer", className="text-white mb-4 text-center"),
            html.Hr(className="text-white"),
            dcc.Link("🏠 Accueil", href="/", className="nav-link", id="link-home"),
            dcc.Link("📊 Monte Carlo", href="/montecarlo", className="nav-link", id="link-montecarlo"),
            dcc.Link("📉 Euler-Maruyama", href="/euler", className="nav-link", id="link-euler"),
            dcc.Link("📈 Vasicek", href="/vasicek", className="nav-link", id="link-vasicek"),
        ], className="p-3")
    ], className="sidebar"),
    
    html.Div(id="page-content", className="main-content")
])

# Callbacks pour la navigation
@app.callback(
    [Output("link-home", "className"),
     Output("link-montecarlo", "className"),
     Output("link-euler", "className"),
     Output("link-vasicek", "className")],
    Input("url", "pathname")
)
def highlight_active_link(pathname):
    classes = ["nav-link"] * 4
    if pathname == "/":
        classes[0] += " active"
    elif pathname == "/montecarlo":
        classes[1] += " active"
    elif pathname == "/euler":
        classes[2] += " active"
    elif pathname == "/vasicek":
        classes[3] += " active"
    return classes

@app.callback(Output("page-content", "children"),
              Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/montecarlo":
        return monte_carlo_layout
    elif pathname == "/euler":
        return euler_layout
    elif pathname == "/vasicek":
        return vasicek_layout
    else:
        return home_layout

# Callbacks pour Monte Carlo
@callback([
    Output("opt-paths", "figure"),
    Output("opt-results", "children")],
    Input("opt-calculate", "n_clicks"),
    [State("opt-S0", "value"),
     State("opt-strike", "value"),
     State("opt-expiry-days", "value"),
     State("opt-rate", "value"),
     State("opt-vol", "value"),
     State("opt-sims", "value"),
     State("opt-type", "value"),
     State("opt-avg-type", "value")])
def update_monte_carlo(n_clicks, S0, K, expiry_days, r, vol, sims, option_type, avg_type):
    if n_clicks is None:
        raise PreventUpdate
    
    # Vérification que toutes les valeurs sont présentes
    if None in [S0, K, expiry_days, r, vol, sims]:
        return go.Figure(), "❌ Veuillez remplir tous les champs obligatoires"
    
    try:
        # Conversion des valeurs
        S0 = float(S0)
        K = float(K)
        expiry_days = int(expiry_days)
        r = float(r) / 100
        vol = float(vol) / 100
        sims = int(sims)
        
        model = MonteCarloPricing(
            S0=S0, K=K, expiry_days=expiry_days,
            r=r, sigma=vol,
            simulations=sims, option_type=option_type, avg_type=avg_type
        )
        paths = model.simulate()
        call, put = model.calculate_prices(paths)

        # Création du graphique
        fig = go.Figure()
        num_traces = min(50, sims)
        for i in range(num_traces):
            fig.add_trace(go.Scatter(
                y=paths[:, i], 
                mode='lines', 
                line=dict(width=1), 
                opacity=0.6,
                showlegend=False
            ))
        
        # Ajout de la moyenne des trajectoires
        mean_path = np.mean(paths, axis=1)
        fig.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Moyenne des trajectoires'
        ))
        
        fig.update_layout(
            title=f"Simulations Monte Carlo - {sims} trajectoires",
            xaxis_title="Jours",
            yaxis_title="Prix de l'actif (€)",
            template="plotly_white",
            height=500
        )

        # Formatage des résultats
        results = html.Div([
            html.H5("Résultats de la simulation:", className="text-primary mb-3"),
            html.P(f"📈 Prix actuel: {model.S0:.2f} €", className="mb-2"),
            html.P(f"🎯 Strike: {model.K:.2f} €", className="mb-2"),
            html.P(f"📅 Jours jusqu'à échéance: {model.steps}", className="mb-2"),
            html.P(f"📊 Volatilité: {model.sigma*100:.2f}%", className="mb-2"),
            html.P(f"💹 Taux sans risque: {model.r*100:.2f}%", className="mb-2"),
            html.Hr(),
            html.H4(f"🟢 Prix CALL: {call:.2f} €", className="text-success mt-3"),
            html.H4(f"🔴 Prix PUT: {put:.2f} €", className="text-danger mt-2"),
            html.Hr(),
            html.P(f"🔢 Simulations: {sims}", className="text-muted mt-3"),
            html.P(f"🎲 Type: {option_type.capitalize()}", className="text-muted")
        ])
        
        return fig, results
        
    except Exception as e:
        return go.Figure(), f"❌ Erreur lors de la simulation: {str(e)}"

# Callbacks pour Euler-Maruyama
@callback([
    Output("euler-paths", "figure"),
    Output("euler-prices", "children")],
    Input("euler-calc", "n_clicks"),
    [State("euler-S0", "value"),
     State("euler-strike", "value"),
     State("euler-expiry-days", "value"),
     State("euler-rate", "value"),
     State("euler-vol", "value"),
     State("euler-sims", "value"),
     State("euler-steps", "value")])
def update_euler(n_clicks, S0, K, expiry_days, r, vol, sims, steps):
    if n_clicks is None:
        raise PreventUpdate
    
    # Vérification que toutes les valeurs sont présentes
    if None in [S0, K, expiry_days, r, vol, sims, steps]:
        return go.Figure(), "❌ Veuillez remplir tous les champs obligatoires"
    
    try:
        # Conversion des valeurs
        S0 = float(S0)
        K = float(K)
        expiry_days = int(expiry_days)
        r = float(r) / 100
        vol = float(vol) / 100
        sims = int(sims)
        steps = int(steps)
        
        model = MonteCarloEuler(
            S0=S0, K=K, expiry_days=expiry_days,
            r=r, sigma=vol,
            simulations=sims, steps=steps
        )
        paths = model.simulate_paths()
        call, put = model.price_option(paths)

        # Création du graphique
        fig = go.Figure()
        num_traces = min(50, sims)
        for i in range(num_traces):
            fig.add_trace(go.Scatter(
                y=paths[:, i], 
                mode='lines', 
                line=dict(width=1), 
                opacity=0.6,
                showlegend=False
            ))
        
        # Ajout de la moyenne des trajectoires
        mean_path = np.mean(paths, axis=1)
        fig.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Moyenne des trajectoires'
        ))
        
        fig.update_layout(
            title=f"Simulations Euler-Maruyama - {sims} trajectoires",
            xaxis_title="Pas de temps",
            yaxis_title="Prix de l'actif (€)",
            template="plotly_white",
            height=500
        )

        # Formatage des résultats
        results = html.Div([
            html.H5("Résultats de la simulation:", className="text-primary mb-3"),
            html.P(f"📈 Prix actuel: {model.S0:.2f} €", className="mb-2"),
            html.P(f"🎯 Strike: {model.K:.2f} €", className="mb-2"),
            html.P(f"📅 Jours jusqu'à échéance: {int(model.T*365)}", className="mb-2"),
            html.P(f"📊 Volatilité: {model.sigma*100:.2f}%", className="mb-2"),
            html.P(f"💹 Taux sans risque: {model.r*100:.2f}%", className="mb-2"),
            html.Hr(),
            html.H4(f"🟢 Prix CALL: {call:.2f} €", className="text-success mt-3"),
            html.H4(f"🔴 Prix PUT: {put:.2f} €", className="text-danger mt-2"),
            html.Hr(),
            html.P(f"🔢 Simulations: {sims}", className="text-muted mt-3"),
            html.P(f"⏱️ Pas de temps: {steps}", className="text-muted")
        ])
        
        return fig, results
        
    except Exception as e:
        return go.Figure(), f"❌ Erreur lors de la simulation: {str(e)}"

# Callbacks pour Vasicek
@callback([
    Output("vasicek-graph", "figure"),
    Output("vasicek-info", "children")],
    Input("vasicek-simulate", "n_clicks"),
    [State("vasicek-r0", "value"),
     State("vasicek-mu", "value"),
     State("vasicek-a", "value"),
     State("vasicek-sigma", "value"),
     State("vasicek-T", "value"),
     State("vasicek-steps", "value"),
     State("vasicek-sims", "value")])
def update_vasicek(n_clicks, r0, mu, a, sigma, T, steps, sims):
    if n_clicks is None:
        raise PreventUpdate
    
    # Vérification que toutes les valeurs sont présentes
    if None in [r0, mu, a, sigma, T, steps, sims]:
        return go.Figure(), "❌ Veuillez remplir tous les champs obligatoires"

    try:
        # Conversion des valeurs
        r0 = float(r0) / 100
        mu = float(mu) / 100
        a = float(a)
        sigma = float(sigma) / 100
        T = float(T)
        steps = int(steps)
        sims = int(sims)

        model = VasicekSimulator(r0, mu, a, sigma, T, steps, sims)
        paths = model.simulate_paths()

        # Création du graphique
        fig = go.Figure()
        num_traces = min(40, sims)
        for i in range(num_traces):
            fig.add_trace(go.Scatter(
                y=paths[:, i] * 100, 
                mode='lines', 
                line=dict(width=1), 
                opacity=0.5,
                showlegend=False
            ))
        
        # Ajout de la moyenne et de la bande de confiance
        mean_path = np.mean(paths, axis=1) * 100
        std_path = np.std(paths, axis=1) * 100
        
        fig.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Moyenne'
        ))
        
        fig.add_trace(go.Scatter(
            y=mean_path + 1.96 * std_path,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            y=mean_path - 1.96 * std_path,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            name='Intervalle de confiance 95%'
        ))
        
        fig.update_layout(
            title=f"Simulations Vasicek - {sims} trajectoires de taux",
            xaxis_title="Pas de temps",
            yaxis_title="Taux d'intérêt (%)",
            template="plotly_white",
            height=500
        )

        info = html.Div([
            html.H5("Paramètres de simulation:", className="text-primary mb-3"),
            html.P(f"📊 Taux initial: {r0*100:.2f}%", className="mb-2"),
            html.P(f"🎯 Moyenne long terme: {mu*100:.2f}%", className="mb-2"),
            html.P(f"⚡ Vitesse de rappel: {a:.2f}", className="mb-2"),
            html.P(f"📈 Volatilité: {sigma*100:.2f}%", className="mb-2"),
            html.P(f"⏰ Maturité: {T} années", className="mb-2"),
            html.Hr(),
            html.P(f"🔢 Simulations: {sims}", className="text-muted mt-3"),
            html.P(f"⏱️ Pas de temps: {steps}", className="text-muted")
        ])
        
        return fig, info
        
    except Exception as e:
        return go.Figure(), f"❌ Erreur lors de la simulation: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=8050)