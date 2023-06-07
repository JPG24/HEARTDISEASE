import numpy as np
import math 

'''
'''
def contineous_treatment(data, t=0):
    y = data[-1,:]
    X = data[:-1,:]
    
    # check for the method we have to use (0 = gain, 1 = gain_ratio)
    match t:
        case 0:
            for row in X:
                # if the column has more than 5 possible results, we have to treat it
                if len(np.unique(row)) > 5:
                    max_gain = -1
                    ps = np.array([])
                    ordre = np.argsort(row)
                    
                    # we search which partition has the better score 
                    y_aux, row_aux = y[ordre], row[ordre]
                    for p in range(len(row) - 1):
                        act_gain,_ = gain(np.array([np.array([row_aux[p], row_aux[p + 1]])]), np.array([y_aux[p], y_aux[p + 1]]))
                        if act_gain[0] > max_gain:
                            max_gain = act_gain[0]
                            ps = np.append(ps, [row_aux[p], row_aux[p + 1]])
                            
                    # we make a binary partition by the best partition found before
                    c = (ps[0] + ps[1]) / 2
                    row[c < row] = 0
                    row[c > row] = 1
        case 1:    
            for row in X:
                # if the column has more than 5 possible results, we have to treat it
                if len(np.unique(row)) > 5:
                    max_gain = 0
                    ordre = np.argsort(row)
                    
                    # we search which partition has the better score
                    y_aux, row_aux = y[ordre], row[ordre]
                    for p in range(len(row) - 1):
                        act_gain,_ = gain_ratio(np.array([np.array([row_aux[p], row_aux[p + 1]])]), np.array([y_aux[p], y_aux[p + 1]]))
                        if act_gain > max_gain:
                            max_gain = act_gain
                            ps = np.array([row_aux[p], row_aux[p + 1]])
                            
                    # we make a binary partition by the best partition found before
                    c = (ps[0] + ps[1]) / 2
                    row[c < row] = 0
                    row[c > row] = 1

    return X.T

def count_possible_labels(dataset):
    result = []
    for column in dataset:
        result.append(np.unique(column)) 
    return np.array(result)

'''
Generate a descicion tree from a data with a specific type of algorithm
    @actual (np.array of np.arrays of floats) is the data that attributes have
    @target (np.array of floats) is y that miss to classificate
    @attributes (np.array of strs) is name tagg of data
    @t is the algorithm what it want to be used:
        0 = ID3 (gain)
        1 = C4.5 (gain_ratio)
        2 = CART (gini)

    return tree (Tree) with the decision tree
'''
def decision_tree(actual, target, attributes, t=0):
    tree = Tree()
    
    # counting how many of each class is in target
    values, count = np.unique(target, return_counts=True)
    target_counts = {}
    for i, key in enumerate(values):
        if key%1 ==0:
            key = int(key)
        target_counts[key] = count[i]
    
    
    # If actual only have one target end branch
    if len(np.unique(target)) == 1:
        tree.create_node(target[0], target_counts, [len(target)])
        return tree
    
    # if all rows of actual are equal means we have to choose the most frequent value
    if (actual == actual[0]).all():        
        # not enough data to categorize all target
        tree.create_node(-1, target_counts, [len(target)])
        return tree
    
    # If actual predicting attributes is empty 
    if actual.shape[0] == 0:
        # not enough data to categorize all target
        tree.create_node(-1, target_counts, [len(target)])
        return tree

    # Select which attribute is the best for the selected type
    match t:
        case 0:
            _, B = gain(actual, target)
        case 1:
            B = gain_ratio(actual, target)
        case 2:
            B = gini(actual, target)
        case default:
            print(f"algorithm {t} isn't in the options")
            return

    # Create a father node with this
    tree.create_node(attributes[B], target_counts, [])

    values_B, count_B = np.unique(actual[B], return_counts=True)
    
    # for each class in the attribute selected, make a new branch from this node
    for val in values_B:
        position = np.where(val == actual[B])[0]
        tree.paste(decision_tree(np.take(np.delete(actual, B, axis=0), position, axis=1), np.take(target, position), np.delete(attributes, B), t=t), val)

    return tree


'''
calculate entropy from a attributes respect to a target
    @actual (np.array of np.arrays of floats) is the data that attributes have
    @target (np.array of floats) is y that miss to classificate

    return S (float) is entropy of the independent array
           all_A (list of lists of floats) contains all entropies of dependent attributes with their probability
'''
def entropy_(actual, target):
    # Calculate target entropy
    # sum(-prob * log_2(prob)) respect target class
    S = entropy_S_A(target)

    # Calculate attribute entropy and their probability
    #sum(-prob * log_2(prob)) respect target class for every class in attribute
    all_A = []
    for A in actual:
        all_A.append(entropy_S_A(target, A))

    return S, all_A

def entropy_S_A(S, A=None):
    if A is not None:
        len_A = len(A)
        class_A = np.unique(A)
        entropy_A = []
        for c in class_A:
            position = np.where(A == c)
            prob = position[0].shape[0] / len_A
            entropy_A.append([entropy_S_A(S[position]), prob])
        return entropy_A

    else:
        len_S = len(S)
        class_S, count_S = np.unique(S, return_counts=True)
        entropy_S = 0
        for c in range(len(class_S)):
            prob = count_S[c] / len_S
            entropy_S -=  prob * math.log(prob, 2)
        return entropy_S

'''
Calculate gain from attributes respect target and give the higher
    @actual (np.array of np.arrays of floats) is the data that attributes have
    @target (np.array of floats) is y that miss to classificate

    return total_gain (np.array of floats) gain of all the classes 
           class_min (int) class with the higher gain
'''
def gain(actual, target):
    S, A = entropy_(actual, target)

    total_gain = np.array([])
    gain_min = float('inf')
    for clss, class_act in enumerate(A):
        gain_act = 0
        for act in class_act:
            gain_act += act[0]*act[1]
        total_gain = np.append(total_gain, S - gain_act)
        if gain_min > gain_act:
            gain_min = gain_act
            class_min = clss
    return total_gain, class_min

'''
Calculate gain ratio from attributes respect target and give the higher
    @actual (np.array of np.arrays of floats) is the data that attributes have
    @target (np.array of floats) is y that miss to classificate

    return class_max (int) class with the higher gain ratio
'''
def gain_ratio(actual, target):
    total_gain, _ = gain(actual, target) 

    max_gain_ratio = 0
    class_max = 0
    for pos, G in enumerate(total_gain):
        act_gain_ratio = G / entropy_S_A(actual[pos])
        if act_gain_ratio > max_gain_ratio:
            max_gain_ratio = act_gain_ratio
            class_max = pos
    return class_max
    
def gini():
    pass
    
'''
Tree class to make decision tree
    @_node (Node) root node 

    def create_node(name[string], target_counts[dict], child_prob[list]) create root node with a name
    def paste(new_tree [Tree], value[float]) paste a tree to root from the branch value
    def show() print tree
    def viz(name[string]) creates a .dot file named "name" with the tree
    def predict(X_test[np.array of np.array of floats], attributes[np.array of strings]) predict for every X_test, y_pred
'''
class Tree:
    def __init__(self) -> None:
        pass

    # create root node
    def create_node(self, name, target_counts, child_prob):
        self._node = Node(name, target_counts, child_prob)

    # paste tree
    def paste(self, new_tree, value):
        self._node.add_child(new_tree._node, value)

    def show(self):
        self._node.show()

    def viz(self, name):
        id = 0
        dot = 'digraph Tree {\nnode [shape=box, style=filled, fillcolor="#FFFFFF"] ;'     
        new_dot, id = self._node.viz(id)
        dot += new_dot + "}"
        with open(f"{name}.dot", "w") as file:
            file.write(dot)

    def predict(self, X_test, attributes):
        y_pred = np.array([])
        # search the i of the first atribute for predicting
        first = np.where(attributes == self._node._name)
        for x in X_test:
            y_pred = np.append(y_pred, self._node.predict(np.take(x, first[0]), x, attributes))
        return y_pred

'''
Node class to make a Tree
    @_name (string) name of the node
    @_childs (list) children nodes
    @_values (np.array) values of each branch
    @_child_prob (list) the number of times that a child is used
    @_target_counts (dict) the total number of every target under this node

    def __init__(name[string], target_counts[dict], child_prob[list]) 
    def add_child(new_node[Node], value[np.float64]) create root node with a name
    def show() print tree
    def viz(name[string]) creates a .dot file named "name" with the tree
    def predict(X_test[np.array of np.array of floats], attributes[np.array of strings]) predict for every X_test, y_pred
'''
class Node:
    def __init__(self, name, target_counts, child_prob) -> None:
        self._childs = []
        self._child_prob = child_prob
        self._target_counts = target_counts
        self._values = np.array([])
        if name == -1:
            if self._target_counts == {}:
                self._name = -1
            else:
                self._name = max(self._target_counts, key=self._target_counts.get)
        else:
            self._name = name
        
        

    def add_child(self, new_node, value):
        self._values = np.insert(self._values, 0, value)
        self._childs.insert(0, new_node)
        self._child_prob.insert(0, sum(new_node._child_prob))

    def show(self, tab=0):
        if tab > 0:
            for t in range(tab - 1):
                print ("    ", end="")
            print ("└── ", end="")
        print (self._name)
        
        if len(self._childs) > 0: 
            tab = tab + 1
            for child in self._childs:
                child.show(tab)

    def viz(self, id, parent_id=False, value=False):
        if type(self._name) == np.str_: 
            dot = f'{id} [color="#F4B80A", label="{self._name}\n{self._target_counts}\n{self._child_prob}"]\n'
        else:
            if self._name % 1 == 0:
                name = int(self._name)
            else:
                name = self._name
                
            if self._name == -1:
                dot = f'{id} [color="#F42D0A", label="{name}\n{self._target_counts}\n{self._child_prob}"]\n'
            else:
                dot = f'{id} [color="#008000", label="{name}\n{self._target_counts}\n{self._child_prob}"]\n'
        actual_id = id
        id += 1
        
        for i, child in enumerate(self._childs):
            new_dot, id = child.viz(id, parent_id=actual_id, value=self._values[i])
            dot += new_dot
            
        if type(parent_id) != bool:
            if value % 1 == 0:
                value = int(value)
            dot += f'{parent_id} -> {actual_id} [labeldistance=2.5, labelangle=45, headlabel="{value}"]\n'
        return dot, id      
        
    
    '''
    @mode is the mode in wich we replace nans or number not founds:
        0 = be child_probs
        1 = by target_probs
    ''' 
    def predict(self, val, x, attributes):
        if len(self._values) == 0:
            return self._name

        next_pos_node = np.where(val == self._values)
        
        if len(next_pos_node[0]) == 0:
            next_pos_node = np.where(self._child_prob == np.amax(self._child_prob))     
                
        if len(self._childs[next_pos_node[0][0]]._values) == 0:     
            return self._childs[next_pos_node[0][0]]._name

        next_val = np.where(self._childs[next_pos_node[0][0]]._name == attributes)
        return self._childs[next_pos_node[0][0]].predict(np.take(x, next_val[0]), x, attributes)








def decision_tree_randomforest(actual, target, attributes, t=0):
    tree = Tree()
    
    # counting how many of each class is in target
    values, count = np.unique(target, return_counts=True)
    target_counts = {}
    for i, key in enumerate(values):
        if key%1 ==0:
            key = int(key)
        target_counts[key] = count[i]
    
    
    # If actual only have one target end branch
    if len(np.unique(target)) == 1:
        tree.create_node(target[0], target_counts, [len(target)])
        return tree
    
    # if all rows of actual are equal means we have to choose the most frequent value
    if (actual == actual[0]).all():        
        # not enough data to categorize all target
        tree.create_node(-1, target_counts, [len(target)])
        return tree
    
    # If actual predicting attributes is empty 
    if actual.shape[0] == 0:
        # not enough data to categorize all target
        tree.create_node(-1, target_counts, [len(target)])
        return tree

    # Select which attribute is the best for the selected type
    match t:
        case 0:
            _, B = gain(actual, target)
        case 1:
            B = gain_ratio(actual, target)
        case 2:
            B = gini(actual, target)
        case default:
            print(f"algorithm {t} isn't in the options")
            return

    # Create a father node with this
    tree.create_node(attributes[B], target_counts, [])

    values_B, count_B = np.unique(actual[B], return_counts=True)
    
    # for each class in the attribute selected, make a new branch from this node
    for val in values_B:
        position = np.where(val == actual[B])[0]
        tree.paste(decision_tree(np.take(np.delete(actual, B, axis=0), position, axis=1), np.take(target, position), np.delete(attributes, B), t=t), val)

    return tree