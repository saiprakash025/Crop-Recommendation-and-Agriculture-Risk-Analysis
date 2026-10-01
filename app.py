import logging

from flask import Flask, make_response, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src import service
from src.config import DEFAULT_HORIZON, FEATURES, SOIL_FEATURES
from src.errors import ModelUnavailableError, UserInputError
from src.forecast import MAX_HORIZON

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

MSG_CURRENT_FAILED = "Crop recommendation could not be generated. Please check your input values and try again."
MSG_FUTURE_FAILED = "Future prediction could not be generated. Please check your input values and try again."
MSG_MODEL_MISSING = "The prediction model is not available on the server right now. Please try again later."
MSG_TOO_LARGE = "The uploaded file is too large. The maximum size is 4 MB."
MSG_SERVER = "Something went wrong on the server. Please try again."

try:
    service.load_artifacts()
except Exception:
    app.logger.exception("Model could not be loaded at startup; it will be retried on the first request")


def is_fetch():
    return request.headers.get("X-Requested-With") == "fetch"


def respond(template, status=200, **context):
    if is_fetch():
        response = make_response(render_template("_results.html", **context), status)
        response.headers["X-Result-Panel"] = "1"
    else:
        response = make_response(render_template(template, **context), status)
    response.headers["Cache-Control"] = "no-store"
    return response


def run_future_safely(soil, horizon, source=None):
    try:
        return service.run_future(soil, horizon, source), None
    except UserInputError as exc:
        app.logger.warning("Future prediction rejected: %s", exc)
        return None, str(exc)
    except ModelUnavailableError:
        app.logger.exception("Model unavailable during future prediction")
        return None, MSG_MODEL_MISSING
    except Exception:
        app.logger.exception("Future prediction failed")
        return None, MSG_FUTURE_FAILED


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "page": "current",
        "values": {**{f: "" for f in FEATURES}, "horizon": str(DEFAULT_HORIZON)},
        "max_horizon": MAX_HORIZON,
        "current": None,
        "future": None,
        "future_error": None,
        "error": None,
    }
    status = 200

    if request.method == "POST":
        try:
            context["values"] = {**{f: request.form.get(f, "") for f in FEATURES},
                                 "horizon": request.form.get("horizon", str(DEFAULT_HORIZON))}
            values = service.parse_inputs(request.form, FEATURES)
            horizon = service.parse_horizon(request.form)

            context["current"] = service.run_current(values)

            soil = {f: values[f] for f in SOIL_FEATURES}
            context["future"], context["future_error"] = run_future_safely(soil, horizon)
        except UserInputError as exc:
            context["error"] = str(exc)
            status = 400
        except RequestEntityTooLarge:
            context["error"] = MSG_TOO_LARGE
            status = 413
        except ModelUnavailableError:
            app.logger.exception("Model unavailable during crop recommendation")
            context["error"] = MSG_MODEL_MISSING
            status = 503
        except Exception:
            app.logger.exception("Crop recommendation failed")
            context["error"] = MSG_CURRENT_FAILED
            status = 500

    return respond("index.html", status, **context)


@app.route("/future", methods=["GET", "POST"])
def future():
    context = {
        "page": "future",
        "values": {**{f: "" for f in SOIL_FEATURES}, "horizon": str(DEFAULT_HORIZON)},
        "max_horizon": MAX_HORIZON,
        "current": None,
        "future": None,
        "future_error": None,
        "error": None,
    }
    status = 200

    if request.method == "POST":
        try:
            context["values"] = {**{f: request.form.get(f, "") for f in SOIL_FEATURES},
                                 "horizon": request.form.get("horizon", str(DEFAULT_HORIZON))}
            soil = service.parse_inputs(request.form, SOIL_FEATURES)
            horizon = service.parse_horizon(request.form)

            upload = request.files.get("history")
            source = upload if upload is not None and upload.filename else None

            context["future"], context["future_error"] = run_future_safely(soil, horizon, source)
            if context["future_error"]:
                if context["future_error"] == MSG_MODEL_MISSING:
                    status = 503
                elif context["future_error"] == MSG_FUTURE_FAILED:
                    status = 500
                else:
                    status = 400
        except UserInputError as exc:
            context["error"] = str(exc)
            status = 400
        except RequestEntityTooLarge:
            context["error"] = MSG_TOO_LARGE
            status = 413
        except Exception:
            app.logger.exception("Future prediction request failed")
            context["error"] = MSG_FUTURE_FAILED
            status = 500

    return respond("future.html", status, **context)


@app.route("/health")
def health():
    return {"status": "ok"}


@app.errorhandler(Exception)
def unhandled_error(exc):
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("Unhandled exception")
    if request.path == "/future":
        page, message = "future", MSG_FUTURE_FAILED
    else:
        page, message = "current", MSG_SERVER
    return respond(
        "future.html" if page == "future" else "index.html",
        500,
        page=page,
        values={"horizon": str(DEFAULT_HORIZON)},
        max_horizon=MAX_HORIZON,
        current=None,
        future=None,
        future_error=None,
        error=message,
    )


if __name__ == "__main__":
    service.load_artifacts(train_if_missing=True)
    app.run(debug=True)
