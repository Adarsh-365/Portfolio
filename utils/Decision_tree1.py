import numpy as np
import pandas as pd

class Node:
    def __init__(self,data):
        self.data = data
        self.children = []
        self.splitter = None
    def print_tree(self,level = 0):
        if self.splitter:
            print(" " * (level * 4) + f"{level}-- " + str(self.data)+" split by "+self.splitter)
        else:
            print(" " * (level * 4) + f"{level}-- " + str(self.data))
            # Indentation based on the level
        for child in self.children:
            child.print_tree(level + 1)  # Recursively print each child

    def add(self,node):
        self.children.append(node)
    
    def predict(self,df):
        if self.splitter:
            for child in self.children:
                if child.data == df[self.splitter]:
                    child.predict(df)
        else:
            print(self.children[0].data)
    
    



class Desiccion_tree:
    def __init__(self,df):
      
       
        self.classfier = df.columns[-1]
        column_list = df.columns[1:-1]
        self.root  =  Node("Null")
        self.Train_(df,column_list,self.root)
        self.class_col  = df[self.classfier].to_list() 
    def print_tree(self):
       self.root.print_tree()
    def predict(self,df):
        return self.root.predict(df)
    def Entropy(self,pos,neg):
      
        if pos == 0 or neg == 0:
            return 0
        else:
            total = pos+neg
            return -(pos/total) *np.log2(pos/total) -(neg/total) *np.log2(neg/total)
    
    def Train_(self,df,column_list,node):
        class_col = df[self.classfier].to_list()
        name,count = np.unique(class_col,return_counts=1)
        print(name,count)
        Entropy_of_tabel = self.Entropy(count[1],count[0])
        
        Entropy_feature = []
        for feature in column_list:
            set2 = []
            feature_col = df[feature].to_list()
            for i in range(len(class_col)):
                set2.append([feature_col[i],class_col[i]])
                
            # print(set2)
            sub_feature = np.unique(df[feature].to_list())
            Entropy_col = []
            for attr in sub_feature:
                pos,neg = 0,0
                for pair in set2:
                    if pair[0]==attr:
                        if pair[1]=="yes":
                            pos+=1
                        else:
                            neg+=1
                
                Entropy_of_attr = ((pos+neg)/len(df))*self.Entropy(pos,neg)
                # print(attr,Entropy_of_attr)
                Entropy_col.append(Entropy_of_attr)
            # print(feature,Entropy_of_tabel-sum(Entropy_col))
            Entropy_feature.append([feature,Entropy_of_tabel-sum(Entropy_col)])
        # print(Entropy_feature)   
        max_val = float("-inf")
        splitting_feature  = None
        for en in Entropy_feature:
            if en[1]>max_val:
                max_val = en[1]
                splitting_feature  = en[0]
        
        print(splitting_feature)
        
        if splitting_feature:
            node.splitter = splitting_feature
            att_of_featrue = np.unique(df[splitting_feature].to_list())
            for sub in att_of_featrue:
                new_df = df[df[splitting_feature]==sub]
                print(new_df)
                sol = np.unique(new_df[self.classfier].to_list())
                if len(sol) ==1:
                    is_classified = True
                else:
                    is_classified = False
                print(sub , "is_classified",is_classified)
                
                if not is_classified:
                    column_list_sub = list(column_list).copy()
                    column_list_sub.remove(splitting_feature)
                    # print(column_list_sub)
                    newnode = Node(sub)
                    node.add(newnode)
                    
                    
                    self.Train_(new_df,column_list_sub,newnode)
                else:
                    newnode = Node(sub)
                    node.add(newnode)
                    
                    node1= Node(sol[0])
                    newnode.add(node1)
                    
                    
            
                
                
                
                # Entropy_of_tabel = self.Entropy(count[1],count[0])
                # print(Entropy_of_tabel)
            
        
        
        
        
data = pd.read_csv("data.csv")

data.columns

Tree = Desiccion_tree(data)
print("")
Tree.print_tree()

df = {
    "age":"senior" ,
    "income":"medium",
    "student":"no",
   "credit rating":"fair"    
}

Tree.predict(df)