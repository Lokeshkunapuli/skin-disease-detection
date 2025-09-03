"""
@author: denil gabani

"""
from flask import render_template, Flask, request, redirect, url_for, flash
import os
from edge_app import pred_at_edge
import time

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret')
UPLOAD_DIR = 'static/data'
os.makedirs(UPLOAD_DIR, exist_ok=True)

SKIN_CLASSES = {
  0: 'akiec, Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'bcc, Basal Cell Carcinoma',
  2: 'bkl, Benign Keratosis',
  3: 'df, Dermatofibroma',
  4: 'mel, Melanoma',
  5: 'nv, Melanocytic Nevi',
  6: 'vasc, Vascular skin lesion'
}

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        start = time.time()
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)
        skin_image = request.files['file']
        if skin_image.filename == '':
            flash('No selected file')
            return redirect(request.url)
        filename = skin_image.filename
        path = os.path.join(UPLOAD_DIR, filename)
        skin_image.save(path)
        disease, accuracy, heatmap_file, backend, sev_level, sev_score, sev_advice = pred_at_edge(path)
        end = time.time()
        if disease.startswith("Error"):
            return render_template(
                "uploaded.html",
                error=disease,
                predictions=None,
                acc=None,
                img_file=skin_image.filename,
                heatmap_file=None,
                backend=None,
                sev_level=None,
                sev_score=None,
                sev_advice=None,
                time_diff=end - start
            )
        else:
            return render_template(
                "uploaded.html",
                title='Success',
                predictions=disease,
                acc=accuracy,
                img_file=skin_image.filename,
                heatmap_file=heatmap_file,
                backend=backend,
                sev_level=sev_level,
                sev_score=sev_score,
                sev_advice=sev_advice,
                time_diff=end - start
            )
    # For GET requests, show the upload page without predictions
    return render_template(
        'uploaded.html',
        title='Upload',
        predictions=None,
        acc=None,
        img_file=None,
        heatmap_file=None,
        backend=None,
        sev_level=None,
        sev_score=None,
        sev_advice=None,
        time_diff=None,
        error=None
    )

if __name__ == "__main__":
    app.run()