set -e

# 1. Create and go INTO the virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install your environment-specific packages
pip install --upgrade pip
pip install ipykernel
pip install -r requirements.txt
python3 -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

# 3. Come OUT of the virtual environment
deactivate

# 4. Install JupyterLab globally
echo "Ensuring JupyterLab is installed globally..."
pip3 install --user jupyterlab

# 5. Fix the PATH variable so the system can find the jupyter command
export PATH="$HOME/.local/bin:$PATH"
export PATH="$(python3 -m site --user-base)/bin:$PATH"

# 6. Launch JupyterLab
jupyter lab --no-browser --port=8888