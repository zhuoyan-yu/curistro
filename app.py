"""Run the maintained local app. The recovered original is tagged dic-2025."""
import os
from curistro import create_app

app = create_app()

if __name__ == '__main__':
    from waitress import serve
    port = int(os.getenv('CURISTRO_PORT', '5058'))
    print(f'Curistro ({app.config["MODE"]} mode): http://127.0.0.1:{port}', flush=True)
    serve(app, host='127.0.0.1', port=port, threads=8)
