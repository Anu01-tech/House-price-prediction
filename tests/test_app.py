"""
Automated tests for the EstateIQ website (uses Streamlit's built-in AppTest).
Run from the project folder:  python tests/test_app.py   (or: pytest tests)
The first run trains the model automatically if artifacts/house_model.joblib is missing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def new_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()


def nav(at, page):
    next(b for b in at.sidebar.button if b.key == f"nav_{page}").click().run()
    assert not at.exception, (page, at.exception)


def field(at, label_start):
    return next(w for w in list(at.number_input) + list(at.slider) if w.label.startswith(label_start))


def submit(at):
    next(b for b in at.button if b.label == "Estimate Price").click().run()


def test_navigation_and_charts():
    at = new_app()
    assert not at.exception
    for page in ["Predict Price", "Model Analytics", "About Project", "Dashboard"]:
        nav(at, page)
    nav(at, "Model Analytics")
    assert len(at.get("plotly_chart")) == 3


def test_prediction_uses_real_model():
    at = new_app()
    nav(at, "Predict Price")
    assert at.session_state.result is None  # nothing shown before submitting
    submit(at)
    low = at.session_state.result["price"]
    field(at, "Median income").set_value(8.0)
    submit(at)
    assert at.session_state.result["price"] > low > 0  # higher income -> higher price


def test_invalid_input_is_rejected():
    at = new_app()
    nav(at, "Predict Price")
    field(at, "Average rooms").set_value(3.0)
    field(at, "Average bedrooms").set_value(5.0)
    submit(at)
    assert at.error and at.session_state.result is None


if __name__ == "__main__":
    test_navigation_and_charts()
    test_prediction_uses_real_model()
    test_invalid_input_is_rejected()
    print("All tests passed")
