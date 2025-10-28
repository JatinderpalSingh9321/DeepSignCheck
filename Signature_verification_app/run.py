from flask import Flask
from app.routes import main
import os

# Absolute path to this directory (SignatureVerificationApp)
base_dir = os.path.abspath(os.path.dirname(__file__))

# Tell Flask exactly where your static and template folders are
app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, "app", "templates"),
    static_folder=os.path.join(base_dir, "app", "static")
)

# Make sure uploads go inside the correct static/uploads directory
upload_folder = os.path.join(app.static_folder, "uploads")
os.makedirs(upload_folder, exist_ok=True)  # Create the folder if it doesn't exist
app.config["UPLOAD_FOLDER"] = upload_folder

# Register your routes
app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)
