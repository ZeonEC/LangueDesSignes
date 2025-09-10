import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.feature import hog

def segment_hand(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    # Fenêtre peau approximative (à ajuster/adapter)
    lower = np.array([0, 133-20, 77-20])
    upper = np.array([255, 133+20, 77+20])
    mask = cv2.inRange(img, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return (mask > 0).astype(np.uint8)

def skeletonize_mask(mask):
    # Plus grande composante
    lab = label(mask)
    if lab.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    largest = np.argmax(np.bincount(lab.flat)[1:]) + 1
    hand = (lab == largest)
    skel = skeletonize(hand)
    # Élagage simple: supprime composantes très petites
    skel = remove_small_objects(skel, min_size=10)
    return skel

def endpoints_and_junctions(skel):
    # Compte voisins 8-connectés pour chaque pixel squelette
    sk = skel.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8); kernel[1,1] = 0
    nbrs = cv2.filter2D(sk, -1, kernel)
    endpoints = np.logical_and(sk==1, nbrs==1)
    junctions = np.logical_and(sk==1, nbrs>=3)
    return np.column_stack(np.where(endpoints)), np.column_stack(np.where(junctions))

def extract_features(img_bgr):
    mask = segment_hand(img_bgr)
    skel = skeletonize_mask(mask)

    # Topo features
    ep, jn = endpoints_and_junctions(skel)
    n_end, n_junc = len(ep), len(jn)

    # Longueurs géodésiques approx. (distance transform sur squelette)
    dist = cv2.distanceTransform((skel==0).astype(np.uint8), cv2.DIST_L2, 3)
    length_est = float(dist[skel].sum() / max(1, skel.sum()))

    # Hu moments sur masque normalisé
    cnts,_ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu = np.zeros(7)
    if cnts:
        cmax = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cmax)
        hu = cv2.HuMoments(M).flatten()

    # HOG sur ROI normalisée 128x128
    roi = cv2.resize((mask*255).astype(np.uint8), (128,128))
    hog_feat = hog(roi, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, block_norm='L2-Hys', visualize=False)

    return np.hstack([n_end, n_junc, length_est, hu, hog_feat])

# Entraînement (X: features, y: labels)
# clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
