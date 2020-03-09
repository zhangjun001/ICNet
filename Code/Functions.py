import SimpleITK as sitk
import numpy as np
import torch.utils.data as Data

def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid

def load_4D(name):
    X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    X = np.reshape(X, (1,)+ X.shape)
    return X

def load_5D(name):
    X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    X = np.reshape(X, (1,)+(1,)+ X.shape)
    return X

def imgnorm(N_I,index1=0.0001,index2=0.0001):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1*len(I_sort))]
    I_max = I_sort[-int(index2*len(I_sort))]
    N_I =1.0*(N_I-I_min)/(I_max-I_min)
    N_I[N_I>1.0]=1.0
    N_I[N_I<0.0]=0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2

def Norm_Zscore(img):
    img= (img-np.mean(img))/np.std(img) 
    return img

def save_img(I_img,savename):
    I2 = sitk.GetImageFromArray(I_img,isVector=False)
    sitk.WriteImage(I2,savename)

def save_flow(I_img,savename):
    I2 = sitk.GetImageFromArray(I_img,isVector=True)
    sitk.WriteImage(I2,savename) 


class Dataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names,iterations,norm=True):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations
  def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names)) [0:2]
        img_A = load_4D(self.names[index_pair[0]])
        img_B = load_4D(self.names[index_pair[1]])        
        if self.norm:
            return  Norm_Zscore(imgnorm(img_A)) , Norm_Zscore(imgnorm(img_B))
        else:
            return img_A, img_B

