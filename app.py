import os

from flask import Flask

from db.schema import ensure_tables
from routes.auth import auth_bp
from routes.emotion_detection import main_bp


def create_app():
    app = Flask(__name__)
    app.secret_key = "change_this_secret_key"

    app.config["CAPTURE_FOLDER"] = os.path.join("static", "captures")
    os.makedirs(app.config["CAPTURE_FOLDER"], exist_ok=True)

    ensure_tables()
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
