import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from skimage import measure
from skimage.segmentation import watershed
# LOAD IMAGE
images = {
    "Bimodal": cv2.imread("bimodal.PNG", 0),
    "Illumination": cv2.imread("iluminating.PNG", 0),
    "Overlapping": cv2.imread("koin.PNG", 0)
}
for name, img in images.items():
    if img is None:
        print(f"ERROR: {name} tidak terbaca")
        exit()
# GROUND TRUTH SEMENTARA
ground_truth = {}
for name, img in images.items():
    _, gt = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ground_truth[name] = gt
# METRIK EVALUASI
def evaluate(gt, pred):
    gt = gt > 0
    pred = pred > 0
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(~gt, ~pred).sum()
    FP = np.logical_and(~gt, pred).sum()
    FN = np.logical_and(gt, ~pred).sum()
    iou  = TP / (TP + FP + FN + 1e-6)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
    acc  = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    prec = TP / (TP + FP + 1e-6)
    rec  = TP / (TP + FN + 1e-6)
    return iou, dice, acc, prec, rec
# OVERLAY
def overlay(img, mask, title):
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (0,0,255), 2)
    plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
# THRESHOLDING
def thresholding(img):
    results = {}
    t = time.time()
    _, res = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    results["Global"] = (res, time.time()-t)
    t = time.time()
    _, res = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    results["Otsu"] = (res, time.time()-t)
    t = time.time()
    res = cv2.adaptiveThreshold(img,255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY,11,2)
    results["Adaptive Mean"] = (res, time.time()-t)
    t = time.time()
    res = cv2.adaptiveThreshold(img,255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,11,2)
    results["Adaptive Gaussian"] = (res, time.time()-t)
    return results
# EDGE DETECTION
def edge_detection(img):
    results = {}
    # Sobel
    t = time.time()
    sx = cv2.Sobel(img, cv2.CV_64F,1,0)
    sy = cv2.Sobel(img, cv2.CV_64F,0,1)
    mag = np.sqrt(sx**2 + sy**2)
    mag = np.uint8(np.clip(mag,0,255))
    _, sob = cv2.threshold(mag,50,255,cv2.THRESH_BINARY)
    results["Sobel"] = (sob, time.time()-t)
    # Prewitt
    t = time.time()
    kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    px = cv2.filter2D(img,-1,kx)
    py = cv2.filter2D(img,-1,ky)
    prew = cv2.add(px,py)
    _, prew = cv2.threshold(prew,50,255,cv2.THRESH_BINARY)
    results["Prewitt"] = (prew,time.time()-t)
    # Canny
    for name,t1,t2 in [
        ("Canny Low",50,100),
        ("Canny Medium",100,200),
        ("Canny High",150,250)]:
        t = time.time()
        can = cv2.Canny(img,t1,t2)
        results[name] = (can,time.time()-t)
    return results
# REGION GROWING
def region_growing(img, seed):
    h,w = img.shape
    visited = np.zeros((h,w),bool)
    result = np.zeros((h,w),np.uint8)
    stack = [seed]
    threshold = 10
    while stack:
        x,y = stack.pop()
        if visited[x,y]:
            continue
        visited[x,y] = True
        result[x,y] = 255
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                nx,ny = x+dx,y+dy
                if 0 <= nx < h and 0 <= ny < w:
                    if not visited[nx,ny]:
                        if abs(int(img[nx,ny])-int(img[x,y])) < threshold:
                            stack.append((nx,ny))
    return result
# WATERSHED
def watershed_seg(img):
    _, thresh = cv2.threshold(img,0,255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(thresh,cv2.DIST_L2,5)
    _, fg = cv2.threshold(dist,0.4*dist.max(),255,0)
    fg = np.uint8(fg)
    markers = measure.label(fg)
    markers = watershed(-dist, markers, mask=thresh)
    return (markers > 1).astype(np.uint8)*255
# CONNECTED COMPONENT
def connected_comp(mask):
    _, labels = cv2.connectedComponents(mask)
    return (labels > 0).astype(np.uint8)*255
# MAIN PROCESS
results = []
for name,img in images.items():
    print("Processing:", name)
    gt = ground_truth[name]
    methods = {}
    methods.update(thresholding(img))
    methods.update(edge_detection(img))
    methods["Region Growing"] = (
        region_growing(img,(img.shape[0]//2,img.shape[1]//2)),0)
    methods["Watershed"] = (watershed_seg(img),0)
    methods["Connected Component"] = (connected_comp(gt),0)
    for method,(mask,t) in methods.items():
        iou,dice,acc,prec,rec = evaluate(gt,mask)
        results.append([
            name,method,iou,dice,acc,prec,rec,t
        ])
# DATAFRAME
df = pd.DataFrame(results, columns=[
    "Image","Method","IoU","Dice",
    "Accuracy","Precision","Recall","Time(s)"
])
summary = df.groupby("Method")[[
    "IoU","Dice","Accuracy","Precision","Recall","Time(s)"
]].mean().sort_values("IoU",ascending=False)
print("\n===== HASIL PER CITRA =====")
print(df.round(4))
print("\n===== RATA-RATA METODE =====")
print(summary.round(4))
# HEATMAP TABEL METRIK (GAMBAR)
plt.figure(figsize=(12,6))
metric_table = summary[[
    "IoU","Dice","Accuracy","Precision","Recall"
]]
sns.heatmap(metric_table,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.5)
plt.title("Tabel Perbandingan Metode Segmentasi (Metrik Evaluasi)")
plt.xlabel("Metric")
plt.ylabel("Method")
plt.tight_layout()
plt.savefig("tabel_metrik_segmentasi.png", dpi=300)
plt.show()
# VISUALISASI SEGMENTASI
for name,img in images.items():
    gt = ground_truth[name]
    methods = thresholding(img)
    methods.update(edge_detection(img))
    methods["Region Growing"] = (region_growing(img,(img.shape[0]//2,img.shape[1]//2)),0)
    methods["Watershed"] = (watershed_seg(img),0)
    methods["Connected Component"] = (connected_comp(gt),0)
    plt.figure(figsize=(18,12))
    plt.subplot(4,4,1)
    plt.imshow(img,cmap='gray')
    plt.title("Original")
    plt.axis("off")
    i = 2
    for m,(mask,_) in methods.items():
        if i > 16:
            break
        plt.subplot(4,4,i)
        plt.imshow(mask,cmap='gray')
        plt.title(m)
        plt.axis("off")
        i += 1
    plt.suptitle(name)
    plt.tight_layout()
    plt.show()
# GRAFIK IoU
plt.figure(figsize=(12,6))
summary["IoU"].sort_values().plot(kind="barh")
plt.title("Perbandingan IoU Semua Metode")
plt.xlabel("IoU")
plt.show()