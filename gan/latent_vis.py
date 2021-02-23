import torch
from train_mnistgan import Generator
from train_mnistcls import Classifier
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

output_dir = "./output"
latent_dim = 100
batch_size = 64
gen_ckpt = "output/G-20.pth"
cls_ckpt = "output/cls-10.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = Generator().to(device)
gen.load_state_dict(torch.load(gen_ckpt))

cls_model = Classifier().to(device)
cls_model.load_state_dict(torch.load(cls_ckpt))

latent_vecs = []
img_cls = []

while len(latent_vecs) < 1000:
    latent_vec = torch.randn((batch_size, latent_dim), device=device)

    gen_img = gen(latent_vec)
    out = cls_model(gen_img)
    pred = out.max(1)[1]
    indx = pred<3
    latent_vecs.extend(latent_vec[indx].detach().cpu().numpy())
    img_cls.extend(pred[indx].detach().cpu().numpy())

latent_vecs = np.array(latent_vecs)
img_cls = np.array(img_cls)

X = latent_vecs
Y = img_cls

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

pca = PCA(n_components=2)
X = pca.fit_transform(X)

model = SVC(kernel='linear')
clf = model.fit(X, Y)

fig, ax = plt.subplots()
title = ('Decision surface of SVC')
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()

np.save(os.path.join(output_dir, "fvecs.npy"), latent_vecs)
np.save(os.path.join(output_dir, "cls.npy"), img_cls)
