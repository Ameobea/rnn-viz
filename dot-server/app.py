from flask import Flask, request, Response
from flask_compress import Compress
from flask_cors import CORS
import subprocess
import graphviz

app = Flask(__name__)
CORS(app)
Compress(app)


@app.route("/dot_to_plainext", methods=["POST"])
def dot_to_plainext():
    dot_input = request.get_data(as_text=True)
    try:
        dot = graphviz.Source(dot_input)
        plainext_output = dot.pipe(format="plain-ext").decode("utf-8")
        return Response(plainext_output, mimetype="text/plain")
    except graphviz.backend.ExecutableNotFound:
        return Response("Graphviz not installed on server.", status=500)
    except graphviz.backend.CalledProcessError:
        return Response("Invalid input dot format.", status=400)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
