"""Independent Dash entrypoint. Streamlit uses app.py / app_EN.py."""
from src.dash_ui.application import create_app

app = create_app()
server = app.server

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=False)
