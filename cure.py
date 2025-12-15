# import library and add constants
import numpy as np
import pandas as pd
import heapq
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

class Cluster:
    
    def __init__(self, id, points_idx, reps, mean):
        """
        @brief Constructor of cluster

        @id (int): unique id of cluster
        @points_idx (list): list of points' index that are inside the cluster
        @reps (list): list of representative points
        @mean: the central point of cluster (not centroid)
        @alive: state of Cluster
        @dist: distance to nearest cluster
        @closest: Id of closest cluster
        """
        self.id = id 
        self.points_idx = [points_idx] if isinstance(points_idx, (int, np.int32, np.int64)) else list(points_idx)
        self.mean = mean
        self.reps = [reps] if not isinstance(reps, list) else reps
        self.alive = True
        self.dist = float("inf")
        self.closest = None

    def __lt__(self, other):
        return self.dist < other.dist

    def to_heap_entry(self):
        return (self.dist, self.id, self)
    
class CURE:
    
    def __init__(self, k, c, alpha):
        """
        k: desired number of clusters
        c: number of representative points per cluster
        alpha: shrink factor (0<=alpha<=1)
        """
        self.k = int(k)
        self.c = int(c)
        self.alpha = float(alpha)

        # these are set on fit()
        self.S = None
        self.n = None
        self.d = None

    def cluster_distance(self, u: Cluster, v: Cluster):
        """
        Distance between two clusters U and V is minimum Euclidean distance between any pair of representative points:
        """
        min_dist = float("inf")
        for p in u.reps:
            for q in v.reps:
                dist = euclidean(p, q)
                if (dist < min_dist):
                    min_dist = dist
        return min_dist

    def calculate_point_before_shrink(self, shrunk_point, mean):
        """ 
        Point is shrunk toward the mean by formula: shrunk_point = original_point + alpha * (mean - original_point).
        Therefore, from shrunk_point, original_point = (shrunk_point - alpha * mean) / (1 - alpha).
        """
        alpha = self.alpha
        if alpha == 1.0:
            return mean
        return (shrunk_point - alpha * mean) / (1.0 - alpha)

    def merge_clusters(self, u: Cluster, v: Cluster, new_id: int):
        """
        @brief Merge two clusters u and v into new cluster w, calculate mean, select new representative points and shrink points.

        @u (Cluster): Cluster u
        @v (Cluster): Cluster v
        @new_id (int): New id of Cluster w
        """
        w_points_idx = u.points_idx + v.points_idx
        w_points = self.S[w_points_idx]
        w_mean = np.mean(w_points, axis = 0)

        repr_selection_set = set()
        for shrunk_point in u.reps:
            original_point = self.calculate_point_before_shrink(shrunk_point, u.mean)
            repr_selection_set.add(tuple(original_point))
            
        for shrunk_point in v.reps:
            original_point = self.calculate_point_before_shrink(shrunk_point, v.mean)
            repr_selection_set.add(tuple(original_point))

        repr_selections = [np.array(point) for point in repr_selection_set]

        # if combine set size is smaller than c --> Not enough repr --> Recalculate repr based on cluster w
        if (len(repr_selections) < self.c):
            repr_selections = w_points

        selected_unshrunk_reps = list()

        # choose first appropriate point
        chosen = None
        max_dist = -1
        for point in repr_selections:
            cur_dist = euclidean(point, w_mean)
            if cur_dist > max_dist:
                chosen = point
                max_dist = cur_dist

        if chosen is not None:
            selected_unshrunk_reps.append(chosen)

        # choose remaining c - 1 points
        for _ in range(self.c - 1):
            max_min_dist = -1
            max_point = None
            for candidate in repr_selections:
                # check if candidate is chose --> skip candidate
                if any(np.array_equal(candidate, selected) for selected in selected_unshrunk_reps):
                    continue

                # find minimum distance to all selected points
                min_dist_to_selected = min(euclidean(candidate, selected) for selected in selected_unshrunk_reps)

                # maximize minimum distance
                if min_dist_to_selected > max_min_dist:
                    max_min_dist = min_dist_to_selected
                    max_point = candidate

            # if new point is found -> insert into list, else all points are inserted, hence break.
            if max_point is not None:
                selected_unshrunk_reps.append(max_point)
            else:
                break

        # shrinking
        w_reps_shrink = list(map(lambda point: point + self.alpha * (w_mean - point), selected_unshrunk_reps))
        
        return Cluster(id = new_id, points_idx = w_points_idx, reps = w_reps_shrink, mean = w_mean)

    def get_rep_data(self, clusters):
        """
        Collects all active shrunk representative points and maps them back to their Cluster ID.
        """
        rep_points = list()
        rep_cluster_map = {} 
        
        for cluster_id, cluster in clusters.items():
            if cluster.alive:
                for rep_point in cluster.reps:
                    rep_cluster_map[tuple(rep_point)] = cluster_id 
                    rep_points.append(rep_point)

        if not rep_points:
            return None, None
        
        return np.array(rep_points), rep_cluster_map

    def build_kd_tree(self, clusters):
        rep_points_array, rep_cluster_map = self.get_rep_data(clusters)
        if rep_points_array is None:
            return None, None
            
        kd_tree = cKDTree(rep_points_array) 
        return kd_tree, rep_cluster_map

    def find_closest_cluster_using_kd_tree(self, query_cluster: Cluster, T, rep_map, threshold_dist=float('inf')):
        """
        Step 15: closest_cluster(T, x, dist(x, w)).
        """
        min_dist = float('inf')
        closest_cluster_id = None
        
        # if T is None:
        #     return None, None
        
        for query_rep in query_cluster.reps:
            # find the nearest neighbor, constrained by threshold
            # use k=2 here to ensure dont accidentally select the query's own point
            distances, indices = T.query(query_rep, k=2, distance_upper_bound=threshold_dist) 
            # if np.isscalar(distances):
            #     distances = np.array([distances])
            #     indices = np.array([indices])
            
            # Iterate over results (up to 2)
            for d_idx, d in enumerate(distances):
                if d >= threshold_dist: # Check against the threshold
                    continue
                # idx = indices[d_idx]
                # if idx >= T.n:
                #     continue
                closest_point = T.data[indices[d_idx]] 
                closest_rep_tuple = tuple(closest_point)
                neighbor_cluster_id = rep_map.get(closest_rep_tuple)
                # Ensure the neighbor is valid (not self)
                if neighbor_cluster_id is not None and neighbor_cluster_id != query_cluster.id:
                    if d < min_dist:
                        min_dist = d
                        closest_cluster_id = neighbor_cluster_id
                        
        # if closest_cluster_id is None:
        #     return None, None
        return closest_cluster_id, min_dist

    def find_closest_neighbor_brute_force(self, u: Cluster, clusters):
        min_dist = float('inf')
        closest_cluster_id = None
        for v_id, v in clusters.items():
            if v_id == u.id or not v.alive:
                continue
            dist = self.cluster_distance(u, v)
            if dist < min_dist:
                min_dist = dist
                closest_cluster_id = v_id
        return closest_cluster_id, min_dist

    def cure(self, S, verbose: bool=False):
        
        # if not isinstance(S, np.ndarray):
        #     S = np.asarray(S)
        # if S.ndim != 2:
        #     raise ValueError("S must be 2D array-like (n_samples, n_features).")
        
        self.S = S
        self.n, self.d = S.shape
        
        clusters = {}
        for i, point in enumerate(self.S):
            # clusters[i] = Cluster(id=i, points_idx=i, reps=point, mean=point.copy())
            clusters[i] = Cluster(id=i, points_idx=i, reps=point, mean=point)

        for i in range(self.n):
            u = clusters[i]
            u.closest, u.dist = self.find_closest_neighbor_brute_force(u, clusters)
        
        # Step 1: T := build_kd_tree(S)
        T, rep_map = self.build_kd_tree(clusters) 

        # Step 2: Q := build_heap(S)
        Q = []
        for cluster_id, u in clusters.items():
            if u.closest is not None:
                heapq.heappush(Q, u.to_heap_entry())

        next_id = self.n

        # Step 3: while size(Q) > k do {
        while len(clusters) > self.k:
            if not Q:
                # if verbose:
                #     print("Heap empty before reaching k clusters; stopping early.")
                break
                
            # Step 4: u := extract_min(Q)
            min_dist, u_id, u_retrieved = heapq.heappop(Q)
            
            if u_id not in clusters or not clusters[u_id].alive:
                continue

            u = clusters[u_id]

            # Step 5: v := u.closest
            v_id = u.closest

            if not clusters.get(v_id) or not clusters[v_id].alive:
                # find for u new closest neighbor, then push back into heap
                u.closest, u.dist = self.find_closest_cluster_using_kd_tree(u, T, rep_map)
                
                if u.closest is None: # if T returns nothing
                    u.closest, u.dist = self.find_closest_neighbor_brute_force(u, clusters)
                if u.closest is not None:
                    heapq.heappush(Q, u.to_heap_entry())
                continue
                
            v = clusters[v_id]
            
            # Step 6: delete(Q, v) handles in step 7
            
            # Step 7: w := merge(u, v)
            w = self.merge_clusters(u, v, next_id)
            
            u.alive = False
            v.alive = False
            del clusters[u_id]
            del clusters[v_id] # Step 6
            clusters[w.id] = w
            
            # Step 8: delete_rep(T, u); delete_rep(T, v); insert_rep(T, w)
            T, rep_map = self.build_kd_tree(clusters) 

            # Step 9: w.closest := x, x random cluster
            w.dist = float('inf') 
            w.closest = None

            # Step 10: for each x in Q do {
            for x_id, x in clusters.items():
                if x_id == w.id or not x.alive:
                    continue
                needs_relocation = False

                # Step 11 - 12
                dist_w_x = self.cluster_distance(w, x)
                if dist_w_x < w.dist:
                    w.closest = x_id
                    w.dist = dist_w_x

                # Step 13. if x.closest is either u or v 
                if x.closest == u_id or x.closest == v_id:
                    dist_x_w = self.cluster_distance(x, w)
                    # Step 15. x.closest := closest_cluster(T, x, dist(x, w)) 
                    # find the true closest neighbor z, constrained by dist(x,w))
                    z_id, dist_x_z = self.find_closest_cluster_using_kd_tree(x, T, rep_map, dist_x_w)
                    
                    # check whether z is better than w
                    if z_id is not None and dist_x_z < dist_x_w:
                        x.closest = z_id
                        x.dist = dist_x_z
                    else: 
                        # Step 17: x.closest := w 
                        x.closest = w.id
                        x.dist = dist_x_w
                    
                    needs_relocation = True

                # Step 20: else if dist(x, x.closest) > dist(x, w) 
                else:
                    dist_x_w = self.cluster_distance(x, w)
                    if dist_x_w < x.dist:
                        # 21. x.closest := w
                        x.closest = w.id
                        x.dist = dist_x_w
                        needs_relocation = True
                
                # Step 18/22: relocate(Q, x)
                if needs_relocation:
                    heapq.heappush(Q, x.to_heap_entry())
                
            # Step 25: insert(Q, w)
            if w.closest is not None:
                heapq.heappush(Q, w.to_heap_entry())
                
            next_id += 1
        
        return list(clusters.values())
    
    def cure_ver2(self, S, verbose: bool=False):
        
        # if not isinstance(S, np.ndarray):
        #     S = np.asarray(S)
        # if S.ndim != 2:
        #     raise ValueError("S must be 2D array-like (n_samples, n_features).")
        
        self.S = S
        self.n, self.d = S.shape
        
        clusters = {}
        for i, point in enumerate(self.S):
            # clusters[i] = Cluster(id=i, points_idx=i, reps=point, mean=point.copy())
            clusters[i] = Cluster(id=i, points_idx=i, reps=point, mean=point)

        for i in range(self.n):
            u = clusters[i]
            u.closest, u.dist = self.find_closest_neighbor_brute_force(u, clusters)
        
        # Step 1: T := build_kd_tree(S)
        T, rep_map = self.build_kd_tree(clusters) 

        # Step 2: Q := build_heap(S)
        Q = []
        for cluster_id, u in clusters.items():
            if u.closest is not None:
                heapq.heappush(Q, u.to_heap_entry())

        next_id = self.n

        # Step 3: while size(Q) > k do {
        while len(clusters) > self.k:
            if not Q:
                # if verbose:
                #     print("Heap empty before reaching k clusters; stopping early.")
                break
                
            # Step 4: u := extract_min(Q)
            min_dist, u_id, u_retrieved = heapq.heappop(Q)
            
            if u_id not in clusters or not clusters[u_id].alive:
                continue

            u = clusters[u_id]

            # Step 5: v := u.closest
            v_id = u.closest

            if not clusters.get(v_id) or not clusters[v_id].alive:
                # find for u new closest neighbor, then push back into heap
                u.closest, u.dist = self.find_closest_cluster_using_kd_tree(u, T, rep_map)
                
                if u.closest is None: # if T returns nothing
                    u.closest, u.dist = self.find_closest_neighbor_brute_force(u, clusters)
                if u.closest is not None:
                    heapq.heappush(Q, u.to_heap_entry())
                continue
                
            v = clusters[v_id]
            
            # Step 6: delete(Q, v) handles in step 7
            
            # Step 7: w := merge(u, v)
            w = self.merge_clusters(u, v, next_id)
            
            u.alive = False
            v.alive = False
            del clusters[u_id]
            del clusters[v_id] # Step 6
            clusters[w.id] = w
            
            # Step 8: delete_rep(T, u); delete_rep(T, v); insert_rep(T, w)
            T, rep_map = self.build_kd_tree(clusters) 

            # Step 9: w.closest := x, x random cluster
            w.dist = float('inf') 
            w.closest = None

            # Step 10: for each x in Q do {
            for x_id, x in clusters.items():
                if x_id == w.id or not x.alive:
                    continue
                needs_relocation = False

                # Step 11 - 12
                dist_w_x = self.cluster_distance(w, x)
                if dist_w_x < w.dist:
                    w.closest = x_id
                    w.dist = dist_w_x

                # Step 13. if x.closest is either u or v 
                if x.closest == u_id or x.closest == v_id:
                    dist_x_w = self.cluster_distance(x, w)
                    # Step 15. x.closest := closest_cluster(T, x, dist(x, w)) 
                    # find the true closest neighbor z, constrained by dist(x,w))
                    z_id, dist_x_z = self.find_closest_cluster_using_kd_tree(x, T, rep_map, dist_x_w)
                    
                    # check whether z is better than w
                    if z_id is not None and dist_x_z < dist_x_w:
                        x.closest = z_id
                        x.dist = dist_x_z
                    else: 
                        # Step 17: x.closest := w 
                        x.closest = w.id
                        x.dist = dist_x_w
                    
                    needs_relocation = True

                # Step 20: else if dist(x, x.closest) > dist(x, w) 
                else:
                    dist_x_w = self.cluster_distance(x, w)
                    if dist_x_w < x.dist:
                        # 21. x.closest := w
                        x.closest = w.id
                        x.dist = dist_x_w
                        needs_relocation = True
                
                # Step 18/22: relocate(Q, x)
                if needs_relocation:
                    heapq.heappush(Q, x.to_heap_entry())
                
            # Step 25: insert(Q, w)
            if w.closest is not None:
                heapq.heappush(Q, w.to_heap_entry())
                
            next_id += 1
        
        return list(clusters.values())

    def get_colors(self, num_clusters):
        cmap = plt.colormaps.get_cmap('viridis')
        return [cmap(i) for i in np.linspace(0, 1, num_clusters)]   

    def plot_2d_clusters(self, labels, title="2D Scatter Plot of Clusters"):
        unique_labels = np.unique(labels)
        colors = self.get_colors(len(unique_labels))
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(unique_labels):
            cluster_points = self.S[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        color=colors[i], label=f'Cluster {label}', 
                        s=50, alpha=0.8)
        plt.title(title)
        plt.xlabel(f'Feature 1 (Axis 0)')
        plt.ylabel(f'Feature 2 (Axis 1)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_3d_clusters(self, labels, title="3D Scatter Plot of Clusters"):
        unique_labels = np.unique(labels)
        colors = self.get_colors(len(unique_labels))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, label in enumerate(unique_labels):
            cluster_points = self.S[labels == label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                    color=colors[i], label=f'Cluster {label}', 
                    s=60, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend()
        plt.show() 

    def plot_high_dim_pca(self, labels, title="PCA Projection (D > 3)"):
        """
        Reduces high-dimensional data (D > 3) to 2 principal components 
        and plots the result.
        """
        pca = PCA(n_components=2)
        S_2d = pca.fit_transform(self.S)

        unique_labels = np.unique(labels)
        colors = self.get_colors(len(unique_labels))

        plt.figure(figsize=(8, 6))
        
        for i, label in enumerate(unique_labels):
            cluster_points = S_2d[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        color=colors[i], label=f'Cluster {label}', 
                        s=50, alpha=0.8)

        explained_variance = pca.explained_variance_ratio_.sum()
        
        plt.title(f"{title}\nExplained Variance: {explained_variance:.2f}")
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_labels_from_clusters(self, final_clusters):
        N = len(self.S)
        labels = np.zeros(N, dtype=int)
        for cluster_id, cluster in enumerate(final_clusters):
            for point_index in cluster.points_idx:
                labels[point_index] = cluster_id
        return labels

    def visualize(self, final_clusters, algorithm_name="CURE"):
        """
        Determines the dimensionality of the data and calls the appropriate 
        visualization function.
        """
        labels = self.get_labels_from_clusters(final_clusters)
        D = self.S.shape[1]
        
        if D == 2:
            self.plot_2d_clusters(labels, title="2D Scatter Plot of Clusters")
        elif D == 3:
            self.plot_3d_clusters(labels, title="3D Scatter Plot of Clusters")
            self.plot_high_dim_pca(labels, f"{algorithm_name} PCA Projection (D={D})")
        elif D > 3:
            self.plot_high_dim_pca(labels, f"{algorithm_name} PCA Projection (D={D})")
        else:
            print("Data dimension is 1. Visualization not implemented.")
            
