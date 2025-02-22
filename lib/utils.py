import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import math
import torch
from scipy.sparse import linalg

class Sphere:
    def __init__(self, dim=2):
        self.dim = dim

    def _is_in_unit_sphere(self, x):
        norm_2 = torch.norm(x, dim=-1)
        return ~(torch.abs(norm_2 - 1) > 1e-7).prod().bool()

    def _ensure_in_unit_sphere(self, x):
        assert self._is_in_unit_sphere(x), 'One of the given vector is not on the unit sphere'

    def _is_in_tangent_space(self, center, v):
        '''
        inputs:
            center: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            if_in_tangence: bool
        '''
        self._ensure_in_unit_sphere(center)
        product = torch.matmul(v, center[:,:,None])
        product[torch.isnan(product)] = 0.0
        return (torch.abs(torch.matmul(v, center[:,:,None])) <= 1e-7).prod().bool()

    def _ensure_in_tangent_space(self, center, v):
        assert self._is_in_tangent_space(center, v), 'One of the given vector is not on the tangent space'

    def _is_in_ctangent_space(self, center, v):
        '''
        inputs:
            center: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            if_in_tangence: bool
        '''
        self._ensure_in_unit_sphere(center)
        v_minus = v[:,:,:-1]
        center_minus = center[:,:-1]
        product = torch.matmul(v_minus, center_minus[:,:,None])
        product[torch.isnan(product)] = 0.0
        return (torch.abs(torch.matmul(v_minus, center_minus[:,:,None])) <= 1e-7).prod().bool()

    def _ensure_in_ctangent_space(self, center, v):
        assert self._is_in_ctangent_space(center, v), 'One of the given vector is not on the cylindrical-tangent space'

    def geo_distance(self, u, v):
        '''
        inputs:
            u: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            induced_distance(u,v): (N, M)
        '''
        assert u.shape[1] == v.shape[2] == self.dim + 1, 'Dimension is not identical.'
        self._ensure_in_unit_sphere(u)
        self._ensure_in_unit_sphere(v)
        return torch.arccos(torch.matmul(v, u[:,:,None]))

    def tangent_space_projector(self, x, v):
        '''
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1)
        outputs:
            project_x(v): (N, M, self.dim + 1)
        '''
        assert x.shape[1] == v.shape[2], 'Dimension is not identical.'

        x_normalized = torch.divide(x, torch.norm(x, dim=-1, keepdim=True))
        v_normalized = torch.divide(v, torch.norm(v, dim=-1, keepdim=True))
        v_on_x_norm = torch.matmul(v_normalized, x_normalized[:,:,None]) #N, M, 1
        v_on_x = v_on_x_norm * x_normalized[:,None,:] #N,M,dim
        p_x = v_normalized - v_on_x #N,M,dim
        return p_x

    def exp_map(self, x, v):
        '''
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) which is on the tangent space of x
        outputs:
            exp_x(v): (N, M, self.dim + 1)
        '''
        assert x.shape[1] == v.shape[2] == self.dim + 1, 'Dimension is not identical.'
        self._ensure_in_unit_sphere(x)
        self._ensure_in_tangent_space(x, v)

        v_norm = torch.norm(v, dim=-1)[:,:,None]  # N,M, 1
        return torch.cos(v_norm) * x[:,None,:] + torch.sin(v_norm) * torch.divide(v, v_norm)

    def log_map(self, x, v):
        '''
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) # v is on the sphere
        outputs:
            log_x(v): (N, M, self.dim + 1)
        '''
        assert x.shape[1] == v.shape[2] == self.dim + 1, 'Dimension is not identical.'
        self._ensure_in_unit_sphere(x)
        self._ensure_in_unit_sphere(v)

        p_x = self.tangent_space_projector(x, v-x[:,None,:]) #N,M,d
        p_x_norm = torch.norm(p_x, dim=-1)[:,:,None] #N,M,1
        distance = self.geo_distance(x, v) #N,M,1
        log_xv = torch.divide(distance * p_x, p_x_norm)
        log_xv[torch.isnan(log_xv)] = 0.0  # map itself to the origin

        return log_xv

    def horizon_map(self, x, v):
        '''
        inputs:
            x: (N, self.dim + 1)
            v: (N, M, self.dim + 1) # v is on the sphere
        outputs:
            H_x(v): (N, M, self.dim + 1)
        '''
        assert x.shape[1] == v.shape[2] == self.dim + 1, 'Dimension is not identical.'
        self._ensure_in_unit_sphere(x)
        self._ensure_in_unit_sphere(v)

        x_minus = x[:,:-1]
        v_minus = v[:,:,:-1]
        p_x_minus = self.tangent_space_projector(x_minus, v_minus - x_minus[:,None,:])
        p_x = torch.cat([p_x_minus, v[:,:,[-1]]- x[:,None,[-1]]], dim=-1)
        p_x_norm = torch.norm(p_x, dim=-1)[:,:,None] 
        distance = self.geo_distance(x, v)
        H_xv = torch.divide(distance * p_x, p_x_norm)
        H_xv[torch.isnan(H_xv)] = 0.0  # map itself to the origin

        return H_xv
    
    def cart3d_to_ctangent_local2d(self, x, v):
        '''
        inputs:
            x: (N, 3)
            v: (N, M, 3) # v is on the ctangent space of x
        outputs:
            \Pi_x(v): (N, M, 2)
        '''
        assert x.shape[1] == v.shape[2] == 3, 'the method can only used for 2d sphere, so the input should be in R^3.'
        self._ensure_in_ctangent_space(x, v)
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        lat, lon = self.xyz2latlon(x1, x2, x3)

        v_temp = v.sum(dim=-1, keepdim=True)
        idx_zero = (v_temp == 0)

        e_phi = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1)
        v_phi = torch.matmul(v, e_phi[:,:,None])
        v_phi[idx_zero] = 0
        v_z = v[:,:,[-1]]
        v_z[idx_zero] = 0
        return torch.cat([v_phi, v_z], dim=-1)

    def cart3d_to_tangent_local2d(self, x, v):
        '''
        inputs:
            x: (N, 3)
            v: (N, M, 3) # v is on the tangent space of x
        outputs:
            \Pi_x(v): (N, M, 2)
        '''
        assert x.shape[1] == v.shape[2] == 3, 'the method can only used for 2d sphere, so the input should be in R^3.'
        self._ensure_in_tangent_space(x, v)
        
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        lat, lon = self.xyz2latlon(x1, x2, x3)
        e_theta = torch.stack([torch.sin(lat)*torch.cos(lon), torch.sin(lat)*torch.sin(lon), torch.cos(lat)], dim=-1) #N,3
        e_phi = torch.stack([-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)], dim=-1) #N,3
        
        v_temp = v.sum(dim=-1, keepdim=True)
        idx_zero = (v_temp == 0)

        v_theta = torch.matmul(v-x[:,None,:], e_theta[:,:,None]) #N,M,1
        v_theta[idx_zero] = 0
        v_phi = torch.matmul(v-x[:,None,:], e_phi[:,:,None]) #N,M,1
        v_phi[idx_zero] = 0
        return torch.cat([v_theta, v_phi], dim=-1)

    @classmethod
    def latlon2xyz(self, lat, lon, is_input_degree=True):
        if is_input_degree == True:
            lat = lat*math.pi/180
            lon = lon*math.pi/180 
        x= torch.cos(lat)*torch.cos(lon)
        y= torch.cos(lat)*torch.sin(lon)
        z= torch.sin(lat)
        return x, y, z

    @classmethod
    def xyz2latlon(self, x, y, z):
        lat = torch.atan2(z, torch.norm(torch.stack([x,y], dim=-1), dim=-1))
        lon = torch.atan2(y, x)
        return lat, lon

class MaxMinScaler:
    """
    Standard the input
    """

    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (self.max - self.min)*data + self.min
    

class KernelGenerator:
    def __init__(self, lonlat, k_neighbors=10, local_map='fast') -> None:
        self.lonlat = lonlat
        self.k_neighbors = k_neighbors
        self.local_map = local_map
        
        self.nbhd_idx, col, row, self.geodesic = self.get_geo_knn_graph(self.lonlat, self.k_neighbors)
        self.sparse_idx = np.array([row, col])
        self.MLP_inputs, self.centers, self.points = self.X2KerInput(self.lonlat, sparse_idx=self.sparse_idx, k_neighbors=self.k_neighbors, local_map=self.local_map)
        _, self.ratio_lists = self.XY2Ratio(self.MLP_inputs[:,-2:], k_neighbors=self.k_neighbors)
        
    def get_geo_knn_graph(self, X, k=25):
        #X: num_node, dim
        lon = X[:, 0]
        lat = X[:, 1]
        x,y,z = latlon2xyz(lat, lon)
        coordinate = np.stack([x,y,z])
        product = np.matmul(coordinate.T, coordinate).clip(min=-1.0, max=1.0) 
        geodesic = np.arccos(product)
        nbhd_idx = np.argsort(geodesic, axis=-1)[:,:k]
        col = nbhd_idx.flatten()
        row = np.expand_dims(np.arange(geodesic.shape[0]), axis=-1).repeat(k, axis=-1).flatten()
        return nbhd_idx, col, row, np.sort(geodesic, axis=-1)[:,:k]

    def X2KerInput(self, x, sparse_idx, k_neighbors, local_map='fast'):
        '''
        x: the location list of each point
        sparse_idx: the sparsity matrix of 2*num_nonzero
        '''
        sample_num = x.shape[0]
        loc_feature_num = x.shape[1]
        centers = x[sparse_idx[0]]
        points = x[sparse_idx[1]]
        if local_map == 'fast':
            delta_x = points - centers
            delta_x[delta_x>180] = delta_x[delta_x>180] - 360
            delta_x[delta_x<-180] = delta_x[delta_x<-180] + 360
            inputs = np.concatenate((centers, delta_x), axis=-1).reshape(-1, loc_feature_num*2)
            inputs = inputs/180*np.pi

        elif local_map == 'log':
            centers = torch.from_numpy(centers.reshape(-1, k_neighbors, loc_feature_num))
            points = torch.from_numpy(points.reshape(-1, k_neighbors, loc_feature_num))
            sphere_2d = Sphere(2)
            centers_x = torch.stack(Sphere.latlon2xyz(centers[:,0,1], centers[:,0,0]), dim=-1)
            points = torch.stack(Sphere.latlon2xyz(points[:,:,1], points[:,:,0]), dim=-1)
            log_cp = sphere_2d.log_map(centers_x, points)
            local_coor = sphere_2d.cart3d_to_tangent_local2d(centers_x, log_cp)
            
            centers = centers.reshape(-1, loc_feature_num).numpy()
            local_coor = local_coor.reshape(-1, loc_feature_num).numpy()
            inputs = np.concatenate((centers/180*np.pi, local_coor), axis=-1).reshape(-1, loc_feature_num*2)

        elif local_map == 'horizon':
            centers = torch.from_numpy(centers.reshape(-1, k_neighbors, loc_feature_num))
            points = torch.from_numpy(points.reshape(-1, k_neighbors, loc_feature_num))
            sphere_2d = Sphere(2)
            centers_x = torch.stack(Sphere.latlon2xyz(centers[:,0,1], centers[:,0,0]), dim=-1)
            points = torch.stack(Sphere.latlon2xyz(points[:,:,1], points[:,:,0]), dim=-1)
            h_cp = sphere_2d.horizon_map(centers_x, points)
            local_coor = sphere_2d.cart3d_to_ctangent_local2d(centers_x, h_cp)
            
            centers = centers.reshape(-1, loc_feature_num).numpy()
            local_coor = local_coor.reshape(-1, loc_feature_num).numpy()
            inputs = np.concatenate((centers/180*np.pi, local_coor), axis=-1).reshape(-1, loc_feature_num*2)
        else:
            raise NotImplementedError('The mapping is not provided.')
        
        return inputs, centers, points  
    
    def XY2Ratio(self, X, k_neighbors=25):
        x = X[:,0]
        y = X[:,1]
        thetas = np.arctan2(y,x)
        thetas = thetas.reshape(-1, k_neighbors)
        ratio_lists = []
        multiples = []
        for theta in thetas:
            theta_unique, counts = np.unique(theta, return_counts=True)
            multiple_list = np.array([theta_unique, counts]).T
            idx = np.argsort(theta_unique)
            multiple_list = multiple_list[idx]
            ratios = []
            ratios_theta = np.zeros_like(theta)
            for i in range(multiple_list.shape[0]):
                if i < multiple_list.shape[0] - 1:
                    ratio = (np.abs(multiple_list[i+1][0] - multiple_list[i][0]) + np.abs(multiple_list[i-1][0] - multiple_list[i][0]))/(2*2*np.pi)
                else: 
                    ratio = (np.abs(multiple_list[0][0]  - multiple_list[i][0]) + np.abs(multiple_list[i-1][0] - multiple_list[i][0]))/(2*2*np.pi)
                ratio = ratio/multiple_list[i][1]
                ratios.append(ratio)
                ratios_theta[theta == multiple_list[i][0]] = ratio
            ratio_lists.append(ratios_theta)
            multiple_list = np.concatenate([multiple_list, np.array([ratios]).T], axis=-1)
            multiples.append(multiple_list)
        return thetas, np.array(ratio_lists)

def latlon2xyz(lat,lon):
    lat = lat*np.pi/180
    lon = lon*np.pi/180 
    x= np.cos(lat)*np.cos(lon)
    y= np.cos(lat)*np.sin(lon)
    z= np.sin(lat)
    return x,y,z

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def nbhd2SparseIdx(nbhds):
    nbhds = np.array(nbhds)
    nbhds_flat = nbhds.flatten()
    element_num = nbhds_flat[nbhds_flat!=-1].shape[0]
    sparse_indices = []

    for nbhd in nbhds:
        nbhd = nbhd[nbhd != -1]
        pairs = np.zeros((nbhd.shape[0], 2))
        pairs[:, 0] = nbhd[0]
        pairs[:, 1] = nbhd
        sparse_indices.append(pairs)
    
    sparse_indices = np.array(sparse_indices)
    sparse_indices = sparse_indices.reshape(element_num, 2)
    sparse_indices = sparse_indices.T
    return torch.LongTensor(sparse_indices)

def latlon2xyz(lat,lon):
    x=-np.cos(lat)*np.cos(lon)
    y=-np.cos(lat)*np.sin(lon)
    z=np.sin(lat)
    return x,y,z

def xyz2latlon(x,y,z):
    lat=np.arcsin(z)
    lon=np.arctan2(-y,-x)
    return lat,lon


if __name__=='__main__':
        lonlat=np.load('/data/zhuxun/xyt/Multi/data_process/lonlat.npy',allow_pickle=True) ### P,2
        kernel_generator = KernelGenerator(lonlat)
        sparse_idx = kernel_generator['sparse_idx']
        print(sparse_idx.shape)