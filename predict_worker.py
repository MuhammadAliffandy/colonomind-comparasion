def isolated_predict(agent_path, scaler_path, img_arr, IMG_SIZE, WAVELET):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    import cv2
    import numpy as np
    import pywt
    import scipy.stats
    import joblib
    import lightgbm as lgb
    
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

    cv2.setNumThreads(0)
    
    gray = cv2.cvtColor(cv2.resize(img_arr, IMG_SIZE), cv2.COLOR_RGB2GRAY)
    coeffs2 = pywt.dwt2(gray, WAVELET)
    LL, (LH, HL, HH) = coeffs2
    def stats(sb):
        return [float(np.mean(sb)), float(np.std(sb)), float(np.var(sb)),
                float(scipy.stats.entropy(np.abs(sb.flatten()) + 1e-6))]
    feats  = stats(LL) + stats(LH) + stats(HL) + stats(HH)
    feats += [float(np.sum(np.square(HH)) / HH.size)]
    
    glcm = graycomatrix(gray, distances=[1,3,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    feats += [
        float(np.mean(graycoprops(glcm, "contrast"))),
        float(np.mean(graycoprops(glcm, "dissimilarity"))),
        float(np.mean(graycoprops(glcm, "homogeneity"))),
    ]
    
    full_feats = [0.5, 0.0, 0.0] + feats
    scaler = joblib.load(scaler_path)
    agent_input = scaler.transform(np.array(full_feats).reshape(1, -1))
    agent = lgb.Booster(model_file=agent_path)
    proba = agent.predict(agent_input)[0]
    
    if not hasattr(proba, "__len__"):
        proba = [float(proba)]
        
    return feats, agent_input.tolist(), list(proba)

if __name__ == "__main__":
    import sys
    import json
    import numpy as np
    
    agent_path = sys.argv[1]
    scaler_path = sys.argv[2]
    img_arr_path = sys.argv[3]
    
    img_arr = np.load(img_arr_path)
    feats, agent_input, proba = isolated_predict(agent_path, scaler_path, img_arr, (224, 224), "db1")
    
    print(json.dumps({"feats": feats, "agent_input": agent_input, "proba": proba}))
