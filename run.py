import sys
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.web.web_server import app, _ensure_models_preloaded

if __name__ == '__main__':
    # Preload models at startup
    _ensure_models_preloaded()
    port = int(os.environ.get('PORT', 8888))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
