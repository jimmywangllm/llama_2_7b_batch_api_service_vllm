#########app_path.py#####
from flask import Flask
from apis import api
import argsparser
args = argsparser.prepare_args()

app = Flask(__name__)

app.config.update(PROPAGATE_EXCEPTIONS=True)
api.init_app(app)

app.run(
	host = '0.0.0.0', 
	port = 3065, 
	use_reloader = False,
	)
#########app_path.py#####