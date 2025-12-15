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
            
            
        
        
# data 

# import numpy as np
# S = np.array([[82.68456706,  4.45922012],
#        [15.86935568, 40.86335396],
#        [61.91947832, 83.62464105],
#        [78.5321037 , 55.72687319],
#        [35.22380287, 69.52947118],
#        [14.31433367, 84.51472094],
#        [45.56225679, 86.70398922],
#        [38.17813064, 36.24561558],
#        [25.12518629, 48.06771104],
#        [38.79577979, 69.8433773 ],
#        [53.89180384,  8.4574877 ],
#        [57.96463358,  2.51607491],
#        [19.87958796, 25.00048336],
#        [29.60111534, 63.85470859],
#        [96.26315396, 75.7944596 ],
#        [45.6781928 , 48.01700787],
#        [47.36043981, 65.33336647],
#        [14.79387894, 53.93647558],
#        [ 0.5349268 , 73.8408741 ],
#        [ 1.9565076 , 38.83379104],
#        [ 7.22259619,  7.46329912],
#        [83.70908182, 87.19282854],
#        [99.46010373, 88.06434118],
#        [10.58629585, 58.17182678],
#        [91.16103317, 81.97991079],
#        [52.33079582, 36.95667037],
#        [16.39406002, 59.02514162],
#        [91.25682157, 47.886654  ],
#        [59.75783353, 99.76065577],
#        [30.12172653, 51.64006229],
#        [87.04887465, 19.54581896],
#        [35.4454102 , 46.78774647],
#        [32.11390573, 81.20586971],
#        [ 4.71182292, 36.74864897],
#        [ 3.56895639, 52.79824126],
#        [20.91179372, 28.06152283],
#        [26.71986957, 88.80691215],
#        [ 9.83678778, 67.92188701],
#        [54.93751982, 43.50173492],
#        [89.75902282, 61.76181973],
#        [68.22378666, 60.52227261],
#        [82.83370095, 52.96478617],
#        [81.74331837, 93.72689515],
#        [16.90881705, 52.60839024],
#        [73.53944709, 69.68580462],
#        [69.43734034, 21.50686882],
#        [15.41377231, 52.82975414],
#        [46.35854535, 53.46012437],
#        [15.79851879, 51.07157605],
#        [16.33483227, 15.27623169],
#        [95.62494752, 65.59904866],
#        [60.98646791, 12.5120598 ],
#        [97.87091774, 60.25955038],
#        [23.25268963, 47.88283382],
#        [66.98653963, 45.82274375],
#        [24.50001518, 12.27595334],
#        [90.62482297, 26.10163377],
#        [26.70519833, 29.97363842],
#        [17.50240288, 16.19866068],
#        [14.62325911, 55.85259812],
#        [53.93148459, 99.97245606],
#        [64.37868097, 93.79043934],
#        [35.21903809, 32.54456791],
#        [56.98641416, 14.63091507],
#        [47.58180387, 82.16518798],
#        [56.53993328, 21.09958963],
#        [89.26654502, 44.93145293],
#        [46.45184173, 37.40672241],
#        [66.12699612, 25.57541456],
#        [29.00718138,  1.33773216],
#        [87.0515256 , 93.26402434],
#        [32.15622127, 78.43082708],
#        [94.28719189,  7.52011228],
#        [14.20634369,  6.98549156],
#        [53.46566368, 75.99886882],
#        [ 9.95331317,  2.71253542],
#        [58.7819291 , 29.15052768],
#        [79.06911821, 15.86612444],
#        [43.31852003, 50.16360299],
#        [58.52516723, 73.98061493],
#        [84.60479974, 47.95630807],
#        [ 4.40048228, 44.45135353],
#        [73.48815099, 21.85374176],
#        [21.2640658 , 15.84101504],
#        [55.26823867, 82.68339427],
#        [86.39822125, 64.03112123],
#        [14.0346749 , 72.51956147],
#        [69.67670801, 17.69164319],
#        [99.25664713, 70.12692082],
#        [35.05569725, 69.25629184],
#        [ 7.65597509, 65.68552145],
#        [51.19585098, 95.57447799],
#        [47.75689988, 19.49903901],
#        [68.53747516,  9.1824247 ],
#        [18.96192323, 18.01509825],
#        [63.17996916, 10.61488992],
#        [68.00907032, 62.11248417],
#        [36.62704183, 72.57986598],
#        [88.28433514, 30.50844673],
#        [84.55027487, 61.67768895],
#        [59.94312233, 12.63362876],
#        [31.69575909, 90.27180413],
#        [65.23166435,  6.20209849],
#        [18.04401365, 67.39479239],
#        [ 6.95013218, 11.01505727],
#        [69.2557347 , 22.88141242],
#        [29.04451722,  8.95596388],
#        [37.3877622 , 37.53720104],
#        [75.32597049, 93.23091484],
#        [28.76642235, 87.88358978],
#        [82.06950295, 98.32890762],
#        [46.51349181, 90.51103712],
#        [61.20806614, 80.32981976],
#        [68.73248806,  2.26841179],
#        [36.30248833, 52.77997883],
#        [32.80198452, 89.83846476],
#        [81.1819152 , 63.65975624],
#        [83.95030926, 12.06409136],
#        [91.51060964, 86.11196053],
#        [16.98574816, 42.20737922],
#        [18.45941299, 91.4280678 ],
#        [ 7.99552013, 20.51448468],
#        [68.05039452, 92.56722853],
#        [26.06795327, 81.69248808],
#        [92.38489181, 21.74713364],
#        [69.94215954, 14.56188632],
#        [19.69490657, 17.37566407],
#        [89.10269625, 39.23804332],
#        [70.92497779, 28.64422886],
#        [91.43405098,  2.5601724 ],
#        [39.32875024, 13.52380743],
#        [88.57709428, 36.94695247],
#        [20.23036818, 98.86408233],
#        [43.84770104, 21.74923664],
#        [46.96982643, 37.32634408],
#        [14.08860497, 36.36306382],
#        [49.39713726, 13.88859656],
#        [ 9.72428221, 41.82910643],
#        [33.55110921, 74.07535433],
#        [19.21161401, 84.95056794],
#        [ 2.05857404, 21.26401099],
#        [65.56571167, 37.06532122],
#        [44.74177903, 72.74861354],
#        [ 5.85473031,  1.0007354 ],
#        [66.62568178, 75.35017042],
#        [46.97446318, 33.20796734],
#        [88.25275129, 47.57305316],
#        [37.16209226, 99.4801409 ],
#        [96.48200633, 99.33198113],
#        [16.24682867, 81.87817515],
#        [60.93794619, 67.17168511],
#        [33.34482629, 99.24595876],
#        [75.50400476,  7.44998757],
#        [41.36792183,  9.77071218],
#        [45.18217333, 43.04981032],
#        [19.9093177 , 79.63131394],
#        [14.14785596, 43.49740298],
#        [30.87305779, 92.15495235],
#        [35.38365041, 58.00158456],
#        [91.65797048, 24.12955327],
#        [62.92176766, 54.11590603],
#        [81.17907883, 81.70157175],
#        [30.16774745, 99.80564807],
#        [95.65703792, 75.65790447],
#        [ 5.15771472, 74.01281953],
#        [21.65343839, 97.08906101],
#        [52.49049315, 86.72663471],
#        [ 6.82667805,  3.75871534],
#        [11.95860806, 86.02196542],
#        [ 4.89411092, 43.47584336],
#        [67.62348882, 11.7502263 ],
#        [46.39875405, 68.02804694],
#        [53.20598591, 34.0089533 ],
#        [95.52395479, 57.24704101],
#        [55.4862418 , 10.95895732],
#        [81.15931526, 46.88664817],
#        [56.69246413, 74.17423445],
#        [98.54556888, 31.02732811],
#        [20.89407524, 22.02683976],
#        [58.11174018, 65.28731387],
#        [31.83236563, 49.23416373],
#        [74.77591598, 47.17096975],
#        [12.02816917, 52.77834219],
#        [ 6.19772266, 29.79945563],
#        [ 2.61194772, 47.8387975 ],
#        [30.94038661, 31.94677585],
#        [76.18204171, 79.27624062],
#        [53.33680115, 74.06711373],
#        [89.87049934, 58.61213401],
#        [87.58609509, 94.56424317],
#        [24.12137522, 37.42993941],
#        [75.24281553,  5.22281284],
#        [15.34360843, 71.91140591],
#        [54.05147419, 92.24689086],
#        [78.74153397, 92.90997195],
#        [63.00926935, 95.19819928],
#        [70.00561018, 78.93404605],
#        [48.7853465 , 58.83323314],
#        [ 4.09693836, 62.35609454],
#        [66.50375609, 40.8561185 ]])


# if __name__ == '__main__':
# model = CURE(k=3, c=2, alpha=0.2)   # same parameters as original default
# clusters = model.cure(S, verbose=True)

# final_clusters = cure_clustering()
# visualize(S, clusters, algorithm_name="CURE algorithm")

# for cluster in clusters:
    # print(cluster.mean)