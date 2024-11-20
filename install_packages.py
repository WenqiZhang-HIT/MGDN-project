import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install pandas
install('pandas')

# install scikit-learn
install('scikit-learn')

# install PyTorch
install('torch')

# install PyTorch Geometric
install('pyg')
