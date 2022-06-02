import joblib
import pickle
import warnings




if 'google.colab' in str(get_ipython()):
    model_dir = "MgAlloyPublic/models"
else:
    model_dir = "models"
    
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=UserWarning)

warnings.filterwarnings('ignore')

models = {"elongation": joblib.load(f"{model_dir}/ductility"),
          "tensile": joblib.load(f"{model_dir}/UTS"),
          "yield": joblib.load(f"{model_dir}/YS")
          }

