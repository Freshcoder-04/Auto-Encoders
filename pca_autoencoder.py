import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self,n_components=2):
        self.dim = n_components

    def fit(self,X):
        self.X = X
        self.mean = np.mean(X,axis = 0)
        self.X_norm = X - self.mean[np.newaxis,:]
        covariances = np.cov(self.X_norm,rowvar=0)
        eigenval, eigenvec = np.linalg.eigh(covariances)
        inds = np.argsort(eigenval)[::-1]
        eigenval = eigenval[inds]
        eigenvec = eigenvec[:,inds]
        self.allcomponents = eigenvec
        self.principal_components = eigenvec[:,:self.dim]

    def encode(self):
        self.X_reduced = np.array(np.dot(self.principal_components.T,self.X_norm.T).T)
        return self.X_reduced

    def forward(self):
        reconstructed = np.dot(self.X_reduced,self.principal_components.T)
        reconstructed += self.mean[np.newaxis,:]
        self.reconstruction_error = np.mean(np.abs(self.X - reconstructed))
        return reconstructed

    def Scree(self):
        self.cumulative_components = np.cumsum(self.allcomponents)/np.sum(self.allcomponents)
        return self.cumulative_components
    
    def ElbowPlot(self):
        rec_errors = []
        for i in range(784):
            self.principal_components = self.allcomponents[:,:i+1]
            _ = self.encode()
            _ = self.forward()
            rec_errors.append(self.reconstruction_error)
            
        plt.figure(figsize=(10, 5))
        plt.plot(rec_errors, label="Reconstruction errors",marker='*')
        plt.xlabel("Number of components")
        plt.ylabel("Reconstruction error")
        plt.title("Reconstruction Error")
        plt.show()