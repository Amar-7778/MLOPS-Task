from flask import Flask, jsonify, render_template, request

from util import (
    FIELD_SECTIONS,
    get_category_levels,
    get_default_values,
    get_model_name,
    predict_learning_outcome,
)


app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        field_sections=FIELD_SECTIONS,
        category_levels=get_category_levels(),
        defaults=get_default_values(),
        model_name=get_model_name(),
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict(flat=True)
    result = predict_learning_outcome(payload)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)