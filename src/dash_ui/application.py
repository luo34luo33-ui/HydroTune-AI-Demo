"""Dash factory; assets belong exclusively to the independent Dash application."""
from pathlib import Path
import os
from secrets import token_hex
from dash import Dash, Input, Output
from flask import request
from .layouts import layout
from .callbacks import register
from .tasks import current


def create_app():
    app=Dash(__name__,title='HydroTune：大语言模型智能体水文平台',
             assets_folder=str(Path(__file__).parent/'assets'),suppress_callback_exceptions=True)
    app.server.secret_key=os.getenv('HYDROTUNE_SESSION_SECRET') or token_hex(32)
    app.server.config.update(SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Lax',MAX_CONTENT_LENGTH=350*1024*1024)
    app.layout=layout
    @app.server.before_request
    def initialize_session():
        if request.path in ('/','/workspace'):
            current()
    register(app)
    app.clientside_callback("""function(path) {
        const media = window.matchMedia('(min-width: 1024px)');
        if (!window.hydroDrawerListener) {
            media.addEventListener('change', function(event) {
                dash_clientside.set_props('configuration-drawer', {open:event.matches});
            });
            window.hydroDrawerListener = true;
        }
        return media.matches;
    }""", Output('configuration-drawer','open'), Input('url','pathname'))
    return app
