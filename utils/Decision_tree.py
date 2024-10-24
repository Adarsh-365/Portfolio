import pandas as pd
import numpy as np
import utils.global_var as gl

# gl ={}
class Node:
    def __init__(self,data):
        self.data  = data
        self.split = None
        self.children =[]
        self.is_leaf = False
    def add(self,node):
        self.children.append(node)
    
    def add_leaf(self,class_val):
        leaf = Node(class_val)
        self.add(leaf)
    
    def print_tree(self, level=0):
        # Prints the tree structure with indentation for levels
        print(" " * level, self.data,self.split)
        
        for child in self.children:
            child.print_tree(level=level+1)
    def predict(self,raw_data):
        # print(self.split)
        if self.split:
            data =  raw_data[self.split][0]
            # print(data,self.split)
            for child in self.children:
                if child.data ==data:
                    return child.predict(raw_data)
        else:
            predication = self.children[0].data[2:-2]
            # print(2,predication)
            return predication
        # self.children[data].predict(raw_data)
        
list_of_entropy ={} 
        


class Decision_tree:
    def __init__(self,csv_file):
        
        self.data = pd.read_csv(csv_file)
        self.class_list = self.data["Class"].to_list()
        self.col_name = self.data.columns[1:-1].to_list()
        gl.feature_list = self.col_name 
        self.root = Node("NULL")
        self.list_of_entropy_of_table = []
        op = self.Calculate_entropy(self.col_name,self.data,self.root)     
        # self.root.print_tree()
    def get_entropy_of_table(self):
        return self.list_of_entropy_of_table
    def predict_data(self,raw_data):
        prediction = self.root.predict(raw_data)
        # print(1,prediction)
        return prediction
    def Entropy(self,P,N,Total):
        if P==0 or N == 0:
            return 0
        else:
            return -(P/Total)*np.log2(P/Total)-(N/Total)*np.log2(N/Total)
        
    def Calculate_entropy(self,col_name,df,root,level = 0):
        
        
        
        values,count = np.unique(self.class_list,return_counts=1)
       # print(values,count)
        if values[0]=='yes':
            pos = count[0]
            neg = count[1]
        else:
            pos = count[1]
            neg = count[0]
        self.Entropy_of_Table = self.Entropy(P=pos,N=neg,Total=pos+neg)
        self.list_of_entropy_of_table.append((self.Entropy_of_Table,pos,neg))
        #print(self.Entropy_of_Table)
        class_list = df["Class"].to_list()
        pair_of_feture_with_entropy = []
        for col in col_name:
            col_list = df[col].to_list()
            pair_set = []
            for i  in range(len(col_list)):
                pair_set.append((col_list[i],class_list[i]))
            #print(pair_set)
            col_uniq = np.unique(col_list)
          #  print(col_uniq)
            entropy_list = []
            entropy_list1 = []
            for cu in col_uniq:
                pos ,neg, total = 0,0,0
                for pair in pair_set:
                    if pair[0] == cu:
                        if pair[1]=='yes':
                            pos+=1
                        else:
                            neg+=1
                total = pos + neg
                single_feture_entropy =(total/len(df))*self.Entropy(pos,neg,total)
                ##############
                if cu not in list_of_entropy:
                      gl.Entropy_list[cu+str(level)] = [pos,neg ,self.Entropy(pos,neg,total) ]
                entropy_list.append(single_feture_entropy)
                entropy_list1.append([self.Entropy(pos,neg,total),total])
                 
          #  print(cu,self.Entropy_of_Table-sum(entropy_list))
            pair_of_feture_with_entropy.append(self.Entropy_of_Table-sum(entropy_list))
            gl.Entropy_list[col+str(level)] = [entropy_list1,sum(entropy_list),col_uniq,len(df)]
        if len(col_name)>0:
            
            split_feature =col_name[pair_of_feture_with_entropy.index(max(pair_of_feture_with_entropy))]
            root.split = split_feature
            item_list = np.unique(df[split_feature].to_list())
            gl.DB_DIVIDE_BY.append(split_feature)
            for item in item_list:
              #  print(item,split_feature)
                temp_col = col_name.copy()
                data = df[df[split_feature] == item]
                
                
                # print(data)
                #df = self.data[self.data[split_feature]==item]
                #print(self.col_name,split_feature,len(np.unique(df["Class"])))
               # print(data)
                # print(split_feature)
                gl.tabel_list[split_feature+str(level)]=data
                if len(np.unique(data["Class"]))!=1:
                    
                    temp_col.remove(split_feature)
                    node = Node(str(item))
                    root.children.append(node)
                    # print(item," is not classified ")
                    
                    self.Calculate_entropy(temp_col,data,node,level+1)
                else:
                    # print(item," is classified ")
                    node = Node(str(item))
                    root.children.append(node)
                    
                    leaf_node = Node(str(np.unique(data["Class"])))
                    leaf_node.is_leaf = True
                    node.children.append(leaf_node)
                    
        return pair_of_feture_with_entropy
                
            
            
            
            
        
        
# tree = Decision_tree("data.csv")
# data = {
#     "RID": [1],
#     "age": ["middle aged"],
#     "income": ["high"],
#     "student": ["no"],
#     "credit rating": ["fair"],
#     "Class": ["no"]
# }
# tree.root.predict(data)
