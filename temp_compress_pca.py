import joblib

path = r"e:/Taneja's Research/Heart-Disease-Detection/artifacts/results/pca_cached.joblib"
obj = joblib.load(path)
joblib.dump(obj, path, compress=3)
print("compressed", path)
