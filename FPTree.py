from collections import defaultdict, Counter, deque
import math
import copy
from pandas import DataFrame as df
import numpy as np
import pandas as pd
import networkx as nx
import os
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import * 
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False




def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color 
class node:
    def __init__(self, item, count, parent_id):
        self.item = item                 #结点的标签也就是某个事务某个中元素
        self.count = count
        '''父亲结点的id 
          在后面FP—tree中insert_node会给每个新的结点分配整型的id，
          记录到  FP_Tree.fp_tree中 '''            
        self.parent_id = parent_id  
        self.child_ids = []      #同理
        
 
class FP_Tree:
   
    def __init__(self, minsup=0.5):
        
        self.minsup = minsup   #最小支持度
        self.minsup_num = None
        self.num = None  # 一个传入数据集的事务数目
 
        self.item_head = defaultdict(list)  # 项头表如{['A']:[1,2,4,5]}由item和对应的结点id list组成  
        self.node_num = 0                # 树的最大节点id
        self.frequent_one_itemsets = defaultdict(lambda: 0)  # 频繁一项集
        self.frequent_k_itemsets = []       # 所有频繁k项集 
        self.frequent_k_itemsets_sup = []   # 所有频繁k项集对应的数目
        self.sort_keys = None               # 用于对每个事务进行排序的依据
        self.G = nx.DiGraph()               # 本tree的networkx图对象，用来画图
        '''
        将树形结构的图转化为字典类型fp_tree，键为节点id，值为node类
        如{[3]:node} 3号结点对应的node，node中本身又保存着父母id，子女id，这样进而表示成tree
          逻辑结构-->存储结构
        '''
        self.fp_tree = defaultdict()
        
        
        
    def ini_param(self, data):
        '''
         使用数据生成一些参数
         data = [[x,x,x],[x,x,x],....]
           
        '''
        self.num = len(data)
        self.minsup_num = math.ceil(self.num * self.minsup)
        self.get_frequent_one_itemsets(data)
        self.build_tree(data)
 
    def get_frequent_one_itemsets(self, data):
        '''
          计算一项集，这一步比较简单
        '''
        c = Counter()
        for t in data:
            c += Counter(t)
        for key, value in c.items():
            if value >= self.minsup_num:
                self.frequent_one_itemsets[key] = value
        #排序
        self.frequent_one_itemsets = dict(sorted(self.frequent_one_itemsets.items(), key=lambda x: x[1], reverse=True))
        # 后续事务排序依据sortkeys
        self.sort_keys = sorted(self.frequent_one_itemsets, key=self.frequent_one_itemsets.get, reverse=True)
        return
 
    def build_tree(self, data):
        one_itemsets = set(self.frequent_one_itemsets.keys())
        self.G.add_node(0,item = 'ROOT',count = 1)
        # 创建根节点
        self.fp_tree[0] = node(item=None, count=0, parent_id=-1)   # 根节点
        self.G.add_node(0,item = 'start',count = 1)
        '''
        扫描整个事务集data，对每个事务排序然后插入到tree中

        '''
        for transaction in data:
            transaction = list(set(transaction) & one_itemsets)  # 去除非频繁项
            if len(transaction) > 0:
                transaction = sorted(transaction, key=self.sort_keys.index)  # 根据排序依据对事务筛进行排序
                parent_id = 0
                #self.G.add_node(0,item = 'start',count = 1)
                for item in transaction:
                    parent_id = self.insert_fptree(parent_id, item) 
            '''
              将item插入到parent_id后，然后返回插入结点的id，
               用于作为下一个item插入的父节点id
               具体看后面insert_fptree注释
            '''
                    
        return
 
    def insert_fptree(self, parent_id, item):
        '''
        插入item：
        如果item没有没有出现在parent_id对应结点的儿子集的item中，
        则创建新结点node，加入到arent_id对应结点的儿子集
        初始画node成员item= item，count=1 等信息  
        然后树的总节点数+1，然后用这数字用来作为此节点的唯一标识id 
        如果存在，直接让对应儿子count+1
        最后返回新结点（或被操作结点id）
        '''
        child_ids = self.fp_tree[parent_id].child_ids
        for child_id in child_ids:
            child_node= self.fp_tree[child_id]
            if child_node.item == item:
                self.fp_tree[child_id].count += 1
                self.G.nodes[child_id]['count'] +=1 
                return child_id
            # if return降低圈复杂度的同时，再判断当前的父节点的子节点中没有项与之匹配，所以新建子节点，更新项头表和树
        self.node_num += 1
        next_node_id = copy.copy(self.node_num)
        self.fp_tree[next_node_id] = node(item=item, count=1, parent_id=parent_id)  # 更新树，添加节点
        self.G.add_node(next_node_id,item = item,count =1)
        self.G.add_edge(parent_id,next_node_id)
        self.fp_tree[parent_id].child_ids.append(next_node_id)  # 更新父节点的孩子列表
        self.item_head[item].append(next_node_id)  # 项头表的建立是和树的建立一并进行的
        return next_node_id
 
    def fit(self, data):
        #构建树
        self.ini_param(data)
        
        ''' 
        现在已经构造好的数据类型有fp树，项头表，频繁一项集。
        现在提取频繁k项集，这时候需要用到项头表里面的节点列表来向上搜索条件FP树，
        后通过条件FP树形成条件模式基，递归得出频繁k项集
        '''
        
        suffix_items_list = []
        suffix_items_id_list = []
        for key, value in self.frequent_one_itemsets.items():
            suffix_items = [key]
            suffix_items_list.append(suffix_items)
            suffix_items_id_list.append(self.item_head[key])
            self.frequent_k_itemsets.append(suffix_items)
            self.frequent_k_itemsets_sup.append(value)
        pre_tree = copy.deepcopy(self.fp_tree)
        self.dfs_search(pre_tree, suffix_items_list, suffix_items_id_list)
        return
 
    def dfs_search(self, pre_tree, suffix_items_list, suffix_items_id_list):
        '''
        suffix_items_list： 后缀元素item集[x,y....]
        suffix_items_id_list：  每个item对应了一些id（一对多）对应的id集
        所以格式为[[idx1,idx2...],[idy1,idy2,idy3].....]
        '''
        for suffix_items, suffix_items_ids in zip(suffix_items_list, suffix_items_id_list):
            '''生成条件树'''
            
            condition_fp_tree = self.get_condition_fp_tree(pre_tree, suffix_items_ids)
            # 根据条件模式基，获取频繁k项集
            new_suffix_items_list, new_suffix_items_id_list = self.extract_frequent_k_itemsets(condition_fp_tree,
                                                                                               suffix_items)
            if new_suffix_items_list:  # 如果后缀有新的项添加进来，则继续递归深度搜索
                # 以开始的单项'G'后缀项为例，经过第一次提取k项频繁集后。单一后缀变为新的后缀项列表[['C', 'G'], ['A', 'G'],
                # ['E', 'G']]，其计数5 5 4也加入到k项集的计数列表里面去了，new_suffix_items_id_list记录了新的后缀项节点id。
                # 此时把原本的pre_tree参数变为条件树，原本的单一后缀项参数变为new_suffix_items_list， 原本的后缀项id列表参数变
                # 为新的id项列表参数。
                # 在这样的递归过程中完成了对k项频繁集的挖掘。
                self.dfs_search(condition_fp_tree, new_suffix_items_list, new_suffix_items_id_list)
        return
 
    def get_condition_fp_tree(self, pre_tree, suffix_items_ids):
        '''
        在pretree中找到suffix_items_ids中一个id的nodes，不断通过parentid生成伸引到根节点的路径
        对每个在suffix_items_ids的id做上述同样的事情最后得到条件树
        condit_tree ：字典  key = 条件树每个结点id，value = 对应的node
        '''
        condition_tree = defaultdict()
        # 从各个后缀叶节点出发，综合各条路径形成条件FP树
        for suffix_items_id in suffix_items_ids:
            suffix_items_count = copy.copy(pre_tree[suffix_items_id].count)
            suffix_items_parent_id = pre_tree[suffix_items_id].parent_id
            #本id搜索路径添加到condition_Tree中
            self.get_path(pre_tree, condition_tree, suffix_items_parent_id, suffix_items_count)
        return condition_tree
 
    def get_path(self, pre_tree, condition_tree, suffix_items_parent_id, suffix_items_count):
        # 递归结束条件：牵引到树根
        if suffix_items_parent_id == 0:
            return
        if suffix_items_parent_id not in condition_tree.keys(): 
            parent_node = copy.deepcopy(pre_tree[suffix_items_parent_id])
            parent_node.count = suffix_items_count
            condition_tree[suffix_items_parent_id] = parent_node
        else:  # 如果叶节点有多个，则肯定是重复路径
            condition_tree[suffix_items_parent_id].count += suffix_items_count
        suffix_items_parent_id = condition_tree[suffix_items_parent_id].parent_id
        self.get_path(pre_tree, condition_tree, suffix_items_parent_id, suffix_items_count)
        return
 
    def extract_frequent_k_itemsets(self, condition_fp_tree, suffix_items):
        ''' 根据条件模式基，提取频繁项集, suffix_item为该条件模式基对应的后缀
           返回新的后缀，以及新添加项(将作为下轮的叶节点)的id,
           不断递归得到不同长度的不同组合的k项集，每次递归一次本函数都会项集长度都会增加1
         '''
        new_suffix_items_list = []          #  后缀中添加的新项
        new_item_head = defaultdict(list)  # 基于当前的条件FP树，更新项头表， 新添加的后缀项
        item_sup_dict = defaultdict(int)
        for key, val in condition_fp_tree.items():
            item_sup_dict[val.item] += val.count  # 对项出现次数进行统计
            new_item_head[val.item].append(key)
 
        for item, sup in item_sup_dict.items():
            if sup >= self.minsup_num:  # 若条件FP树中某个项是频繁的，则添加到后缀中
                current_item_set = [item] + suffix_items
                self.frequent_k_itemsets.append(current_item_set)
                self.frequent_k_itemsets_sup.append(sup)
                new_suffix_items_list.append(current_item_set)
            else:
                new_item_head.pop(item)
        return new_suffix_items_list, new_item_head.values()
    
    def group_by_len(self):
        '''函数名字就是意思……'''
        L = copy.deepcopy(self.frequent_k_itemsets)
        re = {}
        for items in L:
            set_ =  re.get(len(items),set())
            set_.add(frozenset(items))
            re[len(items)] = set_
       
        return sorted(list(re.values()),key= lambda x:len(list(x)[0])  )
        

    
    def get_rules(self, min_conf = 0.5):
      """
      生成关联规则
      用到的变量:
        L:被按长度分组了的k项集
        support_data: 字典. key=  项集  value =  support.
        min_conf: 最小信任度
      返回:
        关联规则: 3元组. 第一项==》第二项 conf= 第三项
      """
      L = copy.deepcopy(self.frequent_k_itemsets)
      L =  list(map(lambda x:frozenset(x),L))
      support_data = dict(zip(L,self.frequent_k_itemsets_sup))
      L = self.group_by_len()
      big_rule_list = []
      sub_set_list = []
      for i in range(0, len(L)):
        for freq_set in L[i]:
          for sub_set in sub_set_list:
            if sub_set.issubset(freq_set):
              conf = support_data[freq_set] / support_data[freq_set - sub_set]
              big_rule = (freq_set - sub_set, sub_set, conf)
              if conf >= min_conf and big_rule not in big_rule_list:
                # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                big_rule_list.append(big_rule)
          sub_set_list.append(freq_set)
      return big_rule_list
   
    
    def show(self,node_size =1800,node_num = 10):
        '''画树  存储结构转又变回逻辑结构的过程...'''
        try:
            asize = 0
            c_ids = self.fp_tree[0].child_ids
            if len(c_ids)==0:
                raise Exception()
                
             
            for id_ in c_ids:
                c_size = self.fp_tree[id_].count
                if asize < c_size:
                    asize = c_size    
            
            self.G.nodes[0]['count']= asize*1.3
        except:   
            asize = self.G.nodes[0]['count']
            print("----------------------------------------------warning：----------------------------------------------------\n\
要保证存在事务中的item中至少存在一个item的支持度是大于支持度。否则只有一个树中只有一个根节点")
            return 
        nodelist  = sorted(list(self.G.nodes(data=True))  ,key= lambda x:x[1]["count"],reverse= True)
        nodelist= nodelist[:min(node_num,len(self.G.nodes))]
        nodelist= [  nodedata[0]  for nodedata in nodelist ]
       
        H = nx.DiGraph(self.G.subgraph(nodelist))
#         plt.figure()
#         nx.draw(H)
#         plt.show()
#         print(type(H))
        labels = { node[0]:node[1]['item']+"\n"+str(node[1]["count"]) for node in H.nodes(data=True) if not node[0] == 0}
        labels[0]= '根'
        labels =pd.Series(labels)
        #labels.index  = range(len(labels))

        size= { node[0]:int(node_size*(node[1]['count']/asize+0.6)) for node in H.nodes(data=True) }
        size = pd.Series(size )
        size.index = range(len(size))

        
        colors=  pd.Series({ node[0]:randomcolor() for node in H.nodes(data=True) })
        
        pos = graphviz_layout(H,prog='dot')
        plt.figure()
        nx.draw(H,pos,with_labels = True, labels = labels, node_color = colors,node_size= size,font_size= 8)
        plt.show()

if __name__ == '__main__':
    # 调试部分  
    print("numpy      :  ",np.__version__)
    print("pandas     :  ",pd.__version__)
    print("networkx   :  ",nx.__version__)
    print("matplotlib :  ",mpl.__version__)
    print("pydot      :  ",pydot.__version__)

    data1 = [list('ABCEFO'), list('ACG'), list('EI'), list('ACDEG'), list('ACEGL'),
                list('EJ'), list('ABCEFP'), list('ACD'), list('ACEGM'), list('ACEGN')]
    tree = FP_Tree(minsup=0.5)
    tree.fit(data1)
        
    print(list(zip(tree.frequent_k_itemsets,tree.frequent_k_itemsets_sup)))
    for item in tree.get_rules(0.75):
        print (item[0], "=>", item[1], "conf: ", item[2])

    tree.G.nodes[0]['count']=8
    tree.show()
    # 运行结果

