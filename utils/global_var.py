import numpy as np
epoch  = 10000
start_button  = None
input_data = np.array([[0,0],
                       [0,1],
                       [1,0],
                       [1,1]])

output = np.array([[0],[1],[1],[0]])
predicted_op = np.array([[0],[0],[0],[0]])
speed = 0.01
start_Neural = False


#########################################
#decsion tree
Entropy_list = {}
feature_list = []
tabel_list = {}

DB_DIVIDE_BY = []


####################################

start_dbscan = False
DBSCAN_Psedo_code =    """
    (DBSCAN ALGORITHM Pseudocode)
    input:
            D : dataset of n object
            e : radius parameter
            Minpits : the Neighbohood Threshold  
    output:
        A set of density based cluster
         
    Method :
        do:
            Randomly select unvisited object pi
            mark pi as visited
            if e-neighborhood of object has at least minpits of object:
                Create cluster C and pi into it
                let N be the number of object in pi e-neighborhood
                for each object p' object in N:
                    if p' is not visited:
                        mark p' visited
                        if e-neighborhood of pi has at least minpits of object:
                            add those object in N
                    if p' is not member of any cluster:
                        add p' in C
                End For
            else:
                mark p as noise
            
            End If
        until:
            no unvisited object is remain
        
            
            
        
    """
dbscan_python =    """
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    # Streamlit app title
    st.title("DBSCAN Clustering with Streamlit")

    # Sidebar for user input
    st.sidebar.header("DBSCAN Parameters")
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 1.0, 0.3, 0.05)
    min_samples = st.sidebar.slider("Minimum Samples", 1, 20, 10, 1)

    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

    # Standardize the features
    X = StandardScaler().fit_transform(X)

    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Display results
    st.write(f'Estimated number of clusters: {n_clusters_}')
    st.write(f'Estimated number of noise points: {n_noise_}')

    # Plot the result
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(10, 6))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        
        # Plot core samples
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        # Plot non-core (border) samples
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title(f'Estimated number of clusters: {n_clusters_}')
    st.pyplot(plt)
    """
