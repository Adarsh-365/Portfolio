
import streamlit as st
import pandas as pd 
import cv2 
import numpy as np
from utils.Decision_tree import Decision_tree,list_of_entropy
import utils.global_var as gl
from utils.neural_network import start_learing,Reset_network



st.title("DATA MINING")
st.write("Mtech 1 Year - 1 sem")
st.write("Teacher - Prof. R. B. V. Subramaanyam")

st.divider()
st.header("Apiriori Algorithmz")
st.divider()
st.header("Hash based Apiriori Algorithm")
st.divider()
st.header("DIC Apiriori Algorithm")
st.divider()
st.header("Pincer Apiriori Algorithm")
st.divider()
st.header("CHARM Apiriori Algorithm")
st.divider()
st.header("Partition Based Apiriori Algorithm")
st.divider()
st.header("Fp Tree Algorithm")
st.divider()
st.header("Decision Tree Algorithm")


uploaded_file = st.file_uploader("CSV  File", type=["csv",])
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.table(df)
    tree = Decision_tree(uploaded_file.name)
    N=5
    st.write("Formula for Gain")
    st.latex(r'''
    H({S}) = - \sum_{{i=1}}^{{n}} p_i \log_2 p_i
''')
    st.latex(r'''  
    Entropy \ of \ Table  \ H({S}) = - \left( \frac{\text{pos}}{\text{total}} \right) \log_2 \left( \frac{\text{pos}}{\text{total}} \right) - \left( \frac{\text{neg}}{\text{total}} \right) \log_2 \left( \frac{\text{neg}}{\text{total}} \right)
''')
    
    list1=tree.get_entropy_of_table()
    # print(list1[0])
    pos = round(list1[0][1],3)
    neg = round(list1[0][2],3)
    total = pos + neg  # Total is the sum of pos and neg
    Entropy_of_tabel = round(list1[0][0],3)
    st.latex(rf'''
    Entropy \ of \ Table  \ H(S) = - \left( \frac{{{pos}}}{{{total}}} \right) \log_2 \left( \frac{{{pos}}}{{{total}}} \right) - \left( \frac{{{neg}}}{{{total}}} \right) \log_2 \left( \frac{{{neg}}}{{{total}}} \right)
''')
    val =0.94
    st.latex(rf'''
    Entropy \ of \ Table  \ H(S) = {Entropy_of_tabel}
    ''')
    
    
    
    
    
    
    # print(list_of_entropy)
    data = {
    "RID": [1],
    "age": ["middle aged"],
    "income": ["high"],
    "student": ["no"],
    "credit rating": ["fair"]
    
    }
    st.table(data)
    
    st.write(tree.predict_data(data))
    # print(gl.feature_list)
    column_counter = 1
    # print(gl.DB_DIVIDE_BY)
    for key in gl.Entropy_list:
        # print(gl.Entropy_list[key])
        if column_counter ==int(str(key)[-1]):
            
            st.latex(rf'''
                From \ Entropy \ of \ different  \ Feature \ Divide database by
            ''')
            
               
                
                
                
                
                
        if len(gl.Entropy_list[key]) ==3:
            pos = gl.Entropy_list[key][0]
            neg = gl.Entropy_list[key][1]
            total = pos + neg
            
            Entropy1 = round( gl.Entropy_list[key][2],3)
            st.latex(rf'''
                Entropy \ of \ {key}  \ H(S) = - \left( \frac{{{pos}}}{{{total}}} \right) \log_2 \left( \frac{{{pos}}}{{{total}}} \right) - \left( \frac{{{neg}}}{{{total}}} \right) \log_2 \left( \frac{{{neg}}}{{{total}}} \right)
            ''')
            
            st.latex(rf'''
            Entropy \ of \ {key}   \ H(S) = {Entropy1}
            ''')
        else:
            pass
            # st.latex(rf'''
            #         Entropy \ of \ {key}  \ H(S) = 
            #     ''')
            
            # Example values for the entropy
            val = str(key)[:-1]
            val1 = str(gl.Entropy_list[key][2][0])
            val2 = str(gl.Entropy_list[key][2][1])
            try:
                val3 = str(gl.Entropy_list[key][2][2])
                latex_string = rf"\text{{Entropy of }} \{{{val}\}} = \text{{Entropy of }} \{{{val1}\}} + \text{{Entropy of }} \{{{val2}\}} + \text{{Entropy of }} \{{{val3}\}}"
           
            except:
                latex_string = rf"\text{{Entropy of }} \{{{val}\}} = \text{{Entropy of }} \{{{val1}\}} + \text{{Entropy of }} \{{{val2}\}} "
           

            # Using f-string to format LaTeX with dynamic values
            st.latex(latex_string)
            
            val = str(key)[:-1]
            gl.tabel_list
            val1 = str(round(gl.Entropy_list[key][0][0][0],2))
            val2 = str(round(gl.Entropy_list[key][0][1][0],2))
            total_col1 = str(round(gl.Entropy_list[key][0][0][1],2))
            total_col2 = str(round(gl.Entropy_list[key][0][1][1],2))
           
            total_df = str(round(gl.Entropy_list[key][3],2))
            
            try:
                total_col3 = str(round(gl.Entropy_list[key][0][2][1],2))
                val3 = str(round(gl.Entropy_list[key][0][2][0],2))
                latex_string = rf"\text{{Entropy of }} \{{{val}\}} = \left( \frac{{{total_col1}}}{{{total_df}}} \right) \{{{val1}\}} + \left( \frac{{{total_col2}}}{{{total_df}}} \right)\{{{val2}\}} +  \left( \frac{{{total_col3}}}{{{total_df}}} \right) \{{{val3}\}}"
           
                
            except:
                latex_string = rf"\text{{Entropy of }} \{{{val}\}} = \left( \frac{{{total_col1}}}{{{total_df}}} \right) \{{{val1}\}} + \left( \frac{{{total_col2}}}{{{total_df}}} \right)\{{{val2}\}} "
           
                
            sum1= str(round(gl.Entropy_list[key][1],2))
            # Using f-string to format LaTeX with dynamic values
            st.latex(latex_string)
            
            entropy_table = "Table"
            entropy_sum = "Sum"

            # Using f-string to format LaTeX with dynamic values
            latex_string = rf"\text{{Entropy }} \{{{val}\}} = \ {sum1} "
            st.latex(latex_string)
            latex_string = rf"\text{{Entropy }} \{{{val}\}} = \text{{Entropy of }} {entropy_table} - \text{{Entropy }} {entropy_sum}"
            st.latex(latex_string)
            latex_string = rf"\text{{Entropy }} \{{{val}\}} =  {Entropy_of_tabel} -  {sum1}"
            st.latex(latex_string)
            latex_string = rf"\text{{Entropy }} \{{{val}\}} =  {round(float(Entropy_of_tabel) -  float(sum1),3)}"
            st.latex(latex_string)
            # print(column_counter,str(key)[-1])
            
            
            
            st.divider()
            
                    
            
            
            #  gl.Entropy_list[col+str(level)] = [self.Entropy_of_Table-sum(entropy_list),sum(entropy_list),col_uniq,1]
    
    
    
        


st.header("CART") 
    

st.divider()
st.header("Random Forest Algorithm")
st.divider()
st.header("SVM Algorithm")
st.divider()



st.header("Neural Network")

def start_Neural_fun(start_button):
    
    gl.start_Neural = True
    

start_button = st.button("Start_Neural")
if start_button:
    
    start_Neural_fun(start_button)
if gl.start_Neural:

    st.subheader("Simple binary opeartion Opearation")
    st.write("Xor Truth Table")


    df = pd.DataFrame({
        'X': [0, 0, 1,1],
        'Y': [0, 1, 0,1],
        'Output':[0, 1, 1,0]
    })

    col1 = st.columns(3)

    with col1[0]:

        st.dataframe(df, hide_index=True)

    with col1[1]:
        
        gl.epoch = st.number_input("Epoch",value=gl.epoch,step=100)
    
        if st.button("Train"):
            
            start_learing()
        if st.button("Reset"):
            
            Reset_network()
            
    # with col1[2]:
    #       gl.speed = st.number_input("Speed",value=gl.speed,step=0.01)
    
                

    image_path = 'XOR1.jpg'  # Path to your existing image
    CV2image = cv2.imread(image_path)
    CV2image = cv2.resize(CV2image,(900,400))

    CV2imageorg = cv2.cvtColor(CV2image, cv2.COLOR_BGR2RGB)
    inpux1_str =f"[{gl.input_data[0][0]},{gl.input_data[1][0]},{gl.input_data[2][0]},{gl.input_data[3][0]}]"
    CV2imageorg= cv2.putText(CV2imageorg, inpux1_str, (10, 90),1, 1, (255, 0, 0), 2)


    output_str =f"[{round(gl.predicted_op[0][0],2)},{round(gl.predicted_op[1][0],2)},{round(gl.predicted_op[2][0],2)},{round(gl.predicted_op[3][0],2)}]"
    CV2imageorg= cv2.putText(CV2imageorg, output_str, (700, 300),1, 1.2, (0, 0, 0), 2)


    inpux2_str =f"[{gl.input_data[0][1]},{gl.input_data[1][1]},{gl.input_data[2][1]},{gl.input_data[3][1]}]"
    CV2imageorg= cv2.putText(CV2imageorg, inpux2_str, (10, 320),1, 1, (255, 0, 0), 2)


    st.image(CV2imageorg, caption='Uploaded Image', use_column_width='auto')
        






st.divider()
st.header("K Mean Clustering")
start_button = st.button("Start K Mean Clustering")

if start_button:
    pass
    st.write("")

st.divider()
st.header("Hierarchical Clustering")
st.divider()
st.header("Birch Clustering")
start_button = st.button("Birch  Clustering")
if start_button:
     st.markdown('# :red[(DBSCAN)] Density-Based Spatial Clustering of Applications with Noise')
st.divider()
st.header("DBSCAN Clustering")

start_button = st.button("Start DBSCAN Clustering")
def start_Dbscan_fun(b):
    if gl.start_dbscan:
        gl.start_dbscan =False
    else:
        gl.start_dbscan =True

if start_button:
    
    start_Dbscan_fun(start_button)



if  gl.start_dbscan:
    pass
   
    st.markdown('# :red[(DBSCAN)] Density-Based Spatial Clustering of Applications with Noise')
    st.markdown('#### - To discover clusters with arbitrary shapes, a density-based method is used.')
    st.markdown('#### - Typically, clusters represent dense regions of objects in data space.')
    st.markdown('#### - The algorithm grows regions with high-density areas into clusters, and the remaining separated parts are called Noise.')
    st.markdown('#### - It defines a cluster as the maximal set of densely connected points.')

    st.markdown('### The following definitions apply:')
    st.markdown('####  - Neighborhood of radius ε of an object is called the :red[ε-neighborhood] of the object.')
    st.markdown('####  - If the ε-neighborhood of an object p has at least MinPts number of objects, then p is called a :red[CORE] point.')
    st.markdown('####  - If an object p is in the ε-neighborhood of q and q is a core point, then p is :red[Directly-Densely-Reachable] from q.')
    st.markdown('####  - p is :red[Densely Reachable] to q if there is a chain of objects like p1...pn, where p1=q and pn=p, and for each pi+1 is Directly-Densely-Reachable from pi where pi belongs to the dataset.')
    st.markdown('####  - p is :red[Densely Connected] to q for ε and MinPts if there is an object o from which p and q are densely reachable for ε and MinPts.')
    st.markdown('####  - Densely Reachable is a transitive closure and asymmetric in nature, meaning only core points are mutually directly reachable. Densely Connected is symmetric in nature.')
        

    # Define code snippets for different languages
   


    cpp_code = """
    // (DBSCAN) Density-Based Spatial Clustering of Applications with Noise
    // C++ Example for DBSCAN
    #include <iostream>
    #include <vector>

    int main() {
        std::vector<std::pair<int, int>> points = {{1, 2}, {2, 2}, {2, 3}, {8, 7}, {8, 8}, {25, 80}};
        for (size_t i = 0; i < points.size(); i++) {
            std::cout << "Point " << i + 1 << ": (" << points[i].first << ", " << points[i].second << ")\\n";
        }
        return 0;
    }
    """

    # Create a radio button for language selection
    language = st.radio("Choose Code Language", ("Python", "Pseudocode", "C++"),horizontal=1)

    # Display the corresponding code based on the selected language
    if language == "Pseudocode":
        st.code(gl.DBSCAN_Psedo_code, language='yaml')
    elif language == "Python":
        st.code(gl.dbscan_python, language='python')
    elif language == "C++":
        st.code(cpp_code, language='cpp')
        # import streamlit as st
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
    eps = st.sidebar.slider("Epsilon (eps)", 0.001, 1.0, 0.3, 0.01)
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
