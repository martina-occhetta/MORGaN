import torch
import pandas as pd
import os
import numpy as np
import networkx as nx

if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    print("device:", device)
else:
    print("CPU")
torch.cuda.empty_cache()


class generate_graph(object):
    def __init__(self, graph_path):
        self.graph_path = graph_path

    def generate_graph(self, file_path, file_name, thr_cpdb, thr_string, thr_domain, thr_exp, thr_go, thr_path, thr_seq):

        file = pd.read_csv(file_path)
        final_gene_node = file['gene'].tolist()          
        print("final_gene_node：", len(final_gene_node))


        print('Generate CDPB_PPI network……')
        ppi = pd.read_csv(os.path.join('Data/multi-view_network/CPDB_ppi.tsv'), sep='\t',
                          compression='gzip', encoding='utf8', usecols=['partner1', 'partner2'])
        ppi.columns = ['source', 'target']                 
        ppi = ppi[ppi['source'] != ppi['target']]         
        ppi.dropna(inplace=True)                                                  
        G = nx.from_pandas_edgelist(ppi)                                       
        ppi_df = nx.to_pandas_adjacency(G)                                       
        temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)           
        ppi_adj = temp.combine_first(ppi_df)                                          
        ppi_adj.fillna(0, inplace=True)                                               
        ppi_cpdb = ppi_adj[final_gene_node].loc[final_gene_node]                    
        ppi_cpdb = ppi_cpdb.applymap(lambda x: 0 if x < thr_cpdb else 1).astype(int)        # Convert the scores to 0 or 1 based on a threshold value
        print(f'The shape of CDPB network_adj：', ppi_cpdb.shape)                     
        ppi_cpdb.to_csv(os.path.join(self.graph_path, str(file_name)+'_cpdb_' + str(thr_cpdb) +'.csv'))
        print("CDPB PPI finished!")


        print('Generate String_PPI network……')
        ppi_score_df = pd.read_csv('Data/multi-view_network/String_ppi.csv')
        ppi_adj_matrix = np.zeros((len(final_gene_node), len(final_gene_node)))        
        gene_index_dict = {gene: index for index, gene in enumerate(final_gene_node)}  
        for _, row in ppi_score_df.iterrows():                                        
            gene1 = row['protein1_name']
            gene2 = row['protein2_name']
            score = row['score']
            # Check if gene1 and gene2 are both present in the final_gene_node
            if gene1 in gene_index_dict and gene2 in gene_index_dict:
                gene1_index = gene_index_dict[gene1]                    
                gene2_index = gene_index_dict[gene2]
                score = np.where(score < thr_string, 0, 1).astype(int)
                ppi_adj_matrix[gene1_index][gene2_index] = score        
                ppi_adj_matrix[gene2_index][gene1_index] = score

        np.fill_diagonal(ppi_adj_matrix, 0)
        ppi_string = pd.DataFrame(ppi_adj_matrix, index=final_gene_node, columns=final_gene_node)
        print(f'The shape of String network_adj：', ppi_string.shape)
        ppi_string.to_csv(os.path.join(self.graph_path, str(file_name)+'_string_' + str(thr_string) + '.csv'))
        print("String PPI finished!")


        print('Generate Domain similarity network……')
        dom_score_df = pd.read_csv('Data/multi-view_network/Domain_similarity.csv')
        domain_matrix = np.zeros((len(final_gene_node), len(final_gene_node)))
        gene_index_dict = {gene: index for index, gene in enumerate(final_gene_node)}
        for _, row in dom_score_df.iterrows():
            gene1 = row['gene1']
            gene2 = row['gene2']
            score = row['score']

            if gene1 in gene_index_dict and gene2 in gene_index_dict:
                gene1_index = gene_index_dict[gene1]
                gene2_index = gene_index_dict[gene2]
                score = np.where(score < 0.3, 0, 1).astype(int)
                domain_matrix[gene1_index][gene2_index] = score
                domain_matrix[gene2_index][gene1_index] = score

        np.fill_diagonal(domain_matrix, 0)
        domain_df = pd.DataFrame(domain_matrix, index=final_gene_node, columns=final_gene_node)
        print(f'The shape of Domain network_adj：', domain_df.shape)
        domain_df.to_csv(os.path.join(self.graph_path, str(file_name) + '_domain_' + str(thr_domain) + '.csv'))
        print("Domain similarity finished!")


        print('Generate four similarity network……')
        exp = pd.read_csv(os.path.join('Data/multi-view_network/Expression_similarity.csv'), sep='\t',
                          index_col=0)                                              
        exp_matrix = exp.applymap(lambda x: 0 if x < thr_exp else 1).astype(int)     
        np.fill_diagonal(exp_matrix.values, 0)

        go = pd.read_csv('Data/multi-view_network/Semantic_similarity.csv', sep='\t',
                         index_col=0)                                             
        go_matrix = go.applymap(lambda x: 0 if x < thr_go else 1).astype(int)      
        np.fill_diagonal(go_matrix.values, 0)

        path = pd.read_csv(os.path.join('Data/multi-view_network/Pathway_similarity.csv'), sep='\t',
                         index_col=0)                                                
        path_matrix = path.applymap(lambda x: 0 if x < thr_path else 1).astype(int)  
        np.fill_diagonal(path_matrix.values, 0)

        seq = pd.read_csv(os.path.join('Data/multi-view_network/Sequence_similarity.csv'), sep='\t',
                          index_col=0)                                               
        seq_matrix = seq.applymap(lambda x: 0 if x < thr_seq else 1).astype(int)     
        np.fill_diagonal(seq_matrix.values, 0)

        networklist = []
        # Iterate through the four similarity matrices
        for matrix in [exp_matrix, go_matrix, path_matrix, seq_matrix]:
            temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)     # Create a temporary data frame called temp
            network = temp.combine_first(matrix)                
            network.fillna(0, inplace=True)  
            network_adj = network[final_gene_node].loc[final_gene_node]             
            networklist.append(network_adj)                                         

        for i, network_adj in enumerate(networklist):
            print(f'The shape of {i+1}th network_adj:', network_adj.shape)          

        networklist[0].to_csv(os.path.join(self.graph_path, str(file_name)+'_exp_' + str(thr_exp) + '.csv'))
        networklist[1].to_csv(os.path.join(self.graph_path, str(file_name)+'_go_' + str(thr_go) + '.csv'))
        networklist[2].to_csv(os.path.join(self.graph_path, str(file_name)+'_path_' + str(thr_path) + '.csv'))
        networklist[3].to_csv(os.path.join(self.graph_path, str(file_name)+'_seq_' + str(thr_seq) + '.csv'))
        print("Four network finished!")

        return ppi_cpdb, ppi_string, domain_df, networklist[0], networklist[1], networklist[2], networklist[3]