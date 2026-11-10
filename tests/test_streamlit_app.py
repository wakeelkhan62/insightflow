from pathlib import Path

from streamlit.testing.v1 import AppTest

import insightflow

APP = Path(insightflow.__file__).parent / "streamlit_app.py"


def test_app_starts_and_asks_for_a_file():
    """The app loads without errors and waits for an upload."""
    app = AppTest.from_file(str(APP)).run()

    assert not app.exception
    assert app.title[0].value == "InsightFlow AI"
    assert "Choose a CSV file" in app.info[0].value
