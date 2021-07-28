import os
import sys
import subprocess
import argparse
import pkg_resources
import re
import time

coll_op_map = {
            "Broadcast": "broadcast_perf",
            "Reduce": "reduce_perf",
            "AllGather": "all_gather_perf",
            "ReduceScatter": "reduce_scatter_perf",
            "AllReduce": "all_reduce_perf",
            "Gather": "gather_perf",
            "Scatter": "scatter_perf",
            "AllToAll": "alltoall_perf",
#            "AllToAllv": "alltoallv_perf",
            "Send": "sendrecv_perf",
            "Recv": "sendrecv_perf",
            }

reduction_op_map = {
                "0" : "sum",
                "1" : "prod",
                "2" : "max",
                "3" : "min",
                "4" : "all",
               }

data_types_map = {
                "0" : "int8",
                "1" : "uint8",
                "2" : "int32",
                "3" : "uint32",
                "4" : "int64",
                "5" : "uint64",
                "6" : "half",
                "7" : "float",
                "8" : "double",
                "9" : "bf16",
                #"10" : "ncclNumTypes Equivalent?"
             }

data_type_bytes_map = {
                    "0" : 1,
                    "1" : 1,
                    "2" : 4,
                    "3" : 4,
                    "4" : 8,
                    "5" : 8,
                    "6" : 2,
                    "7" : 4,
                    "8" : 8,
                    "9" : 2,
                    #"10" : Not sure.
                  }

# n: number of ranks 
# n links of Bandwidth B to perform a operation
# def factor_1(n): 
#     return n
def all_gather_factor(n):
     return (n-1)/n
def reduce_scatter_factor(n):
     return (n-1)/n        
def all_reduce_factor(n):
     return 2*(n-1)/n
def all_to_all_factor(n):
    return 2*(n-1)/n

def coll_op_algobw_factor(coll_type, nranks):
    nranks = float(nranks)
    if coll_type == "AllGather":
        return float(all_gather_factor(nranks))
    elif coll_type == "ReduceScatter":
        return float(reduce_scatter_factor(nranks))
    elif coll_type == "AllReduce":
        return float(all_reduce_factor(nranks))
    elif coll_type == "AllToAll":
        return float(all_to_all_factor(nranks))
    else:
        return float(1)
    
def algobw_factor_times_size(coll_type, nranks, total_bytes):
    nranks = float(nranks)
    if coll_type == "AllGather":
        return all_gather_factor(nranks) * float(total_bytes)
    elif coll_type == "ReduceScatter":
        return reduce_scatter_factor(nranks) * float(total_bytes)
    elif coll_type == "AllReduce":
        return all_reduce_factor(nranks) * float(total_bytes)
    elif coll_type == "AllToAll":
        return all_to_all_factor(nranks) * float(total_bytes)
    else:
        return float(1) * float(total_bytes)   

class DisjointSet(object): # https://stackoverflow.com/questions/3067529/a-set-union-find-algorithm
    def __init__(self):
        self.leader = {} # maps a member to the group's leader
        self.group = {} # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

                
def get_useful_info(log_file):
    fs = open(log_file, 'r')
    lines = fs.readlines()
    fs.close()

    coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines = [], [], [], [], [], []
    for j in range(len(lines)):
        line = lines[j].rstrip()
        if ("opCount" in line and "sendbuff" in line):
            coll_lines.append(line)
        elif ("Channel" in line and "via" in line):
            conn_lines.append(line)
        elif ("Init COMPLETE" in line and "busId" in line):
            comm_lines.append(line)
        elif ("NCCL INFO Ring" in line):
            ring_lines.append(line)
        elif ("NCCL INFO Trees" in line):
            tree_lines.append(line)
        elif ((" ## " in line) and ("KL HWID" in line or "KE" in line or "CE" in line)):  # RCCL From ROCm 4.3
            # Bug: [ 6628.064978] we need to consider the case when there is a spac right after '['
            #                     Everything with split_list[index] will break.
            if "[ " in line:
                line = line.replace("[ ", "[")
            coll_trace_lines.append(line)
            
    return coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines

# def coll_table(coll_lines):
#     opCount, coll, count, datatype, op_type, root, comm, nnranks = [], [], [], [], [], [], [], []
#     for line in coll_lines:
#         split_list = line.split(" ")
#         coll.append(split_list[4][:-1])
#         opCount.append(split_list[6])
#         count.append(split_list[12])
#         datatype.append(split_list[14])
#         op_type.append(split_list[16])
#         root.append(split_list[18])
#         comm.append(split_list[20])
#         nnranks.append(next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", ""))

#     dict_coll = {'coll': coll, 'opCount': opCount, 'datatype': datatype, 'count':count, 'op_type':op_type, 'root':root, 'comm':comm, 'nnranks':nnranks}    
#     return pd.DataFrame(dict_coll)

def coll_table(coll_lines):
    opCount, coll, count, datatype, op_type, root, comm, nranks, data_size = [], [], [], [], [], [], [], [], []
    for line in coll_lines:
        split_list = line.split(" ")
        coll.append(split_list[4][:-1])
        opCount.append(int(split_list[6], 16))
        count.append(split_list[12])
        datatype.append(split_list[14])
        op_type.append(split_list[16])
        root.append(split_list[18])
        comm.append(split_list[20])
        nranks.append(int(next(item for item in split_list if 'nranks' in item).split("=")[1].replace("]", ""))) ### 
        data_size.append(int(split_list[split_list.index("count") + 1]) * data_type_bytes_map[split_list[split_list.index("datatype") + 1]])
        
    dict_coll = {'coll': coll, 'opCount': opCount, 'datatype': datatype, 'count':count, 'op_type':op_type, 'root':root, 'comm':comm, 'nranks':nranks, 'data_size':data_size}
    return pd.DataFrame(dict_coll)
    # new_table['algobw_factor_times_size'] = new_table.apply(lambda row: algobw_factor_times_size(row['coll'], row['nranks'], row['data_size']), axis=1)


## It works for connection information like XXX via "P2P/direct pointer%s".(useReadStr),       --> pointer%s will not be collected
##                                                  "P2P/IPC%s".(useReadStr),  
##                                                  "P2P/indirect/%d[%lx]%s".(intermediateRank,comm->peerInfo[intermediateRank].busId, useReadStr) 
##                                                  "direct shared memory"                     --> shared memory
## However, it will not capture the information for CollNet for now.

## xxx [4] NCCL INFO Channel 00 : 0[e3000] -> 1[c3000] via P2P/IPC comm 0x7f53bc000e50 nRanks 04',  #(new output)
def conn_table_modified(conn_lines):
    def process_string(line):
        split_list = line.split("[")
        return [split_list[0], split_list[1].split("]")[0]]

    start_rank, start_busid, end_rank, end_busid, connection, comm, nranks = [], [], [], [], [], [], []
    for line in conn_lines:
        split_list = line.split(" ")
        sr, sb = process_string(split_list[split_list.index(":") + 1])  # first device
        er, eb = process_string(split_list[split_list.index("->") + 1]) # second device
        start_rank.append(sr)
        start_busid.append(sb)
        end_rank.append(er)
        end_busid.append(eb)
        connection.append(split_list[split_list.index("via") + 1])  # if it is direct, it means the connection is done by direct shared memory
        comm.append(split_list[split_list.index("comm") + 1])
        nranks.append(int(split_list[split_list.index("nRanks") + 1]))

    dict_conn = {'start_rank': start_rank, 'start_busid': start_busid, 'end_rank': end_rank, 'end_busid': end_busid, 
                 'connection': connection, 'comm':comm, 'nranks':nranks, 'conn_line':conn_lines}      
    return pd.DataFrame(dict_conn)


def comm_table(comm_lines):
    comm, rank, nranks, cudaDev, busId = [], [], [], [], []   
    for line in comm_lines:
        split_list = line.rstrip().split(" ")
        comm.append(split_list[5])
        rank.append(split_list[7])
        nranks.append(int(split_list[9]))
        cudaDev.append(split_list[11])
        busId.append(split_list[13])
    dict_comm = {'comm':comm, 'rank':rank, 'nranks':nranks, 'cudaDev':cudaDev, 'busId':busId}  
    return pd.DataFrame(dict_comm)


def topo_table(topo_lines):
    comm, nranks, busId, topo_line = [], [], [], []
    for line in topo_lines:
        split_list = line.split(" ")
        comm.append(split_list[split_list.index("comm") + 1])
        nranks.append(int(split_list[split_list.index("nRanks") + 1]))
        busId.append(split_list[split_list.index("busId") + 1])
        topo_line.append(line)
    dict_table = {'comm':comm, 'nranks':nranks, 'busId':busId, 'topo_line':topo_line}      
    return pd.DataFrame(dict_table)

def create_table(log_name):
    log_file = os.path.abspath(log_name)
    coll_lines, conn_lines, comm_lines, ring_lines, tree_lines, coll_trace_lines = get_useful_info(log_file)
    return coll_table(coll_lines), conn_table_modified(conn_lines), comm_table(comm_lines), topo_table(ring_lines), topo_table(tree_lines), coll_trace_lines


def device_grouping(comm_table, conn_table):
    groups = []
    for index, row in comm_table.iterrows():
        temp = [row['busId'], list(conn_table[(conn_table['comm'] == row['comm']) & (conn_table['start_busid'] == row['busId'])]['end_busid'].unique())]
        groups.append(temp)    
    nranks = list(comm_table['nranks'])
    outputs = []
    rank_outputs = []
    tempRank = None
    for id, group in enumerate(groups): 
        if tempRank == None:
            tempRank = nranks[id]
            ds = DisjointSet()
        else:
            if tempRank != nranks[id]:
                for _, v in ds.group.items():
                    if v not in outputs: 
                        outputs.append(v)
                ds = DisjointSet()
                tempRank = nranks[id]
        for node in group[1]:
            ds.add(group[0], node)

        if id == len(groups) - 1: 
            for _, v in ds.group.items():
                if v not in outputs: 
                    outputs.append(v)  
    return outputs

def collect_topo(comm, nranks, ring_table, tree_table, conn_table, lines):
    ring_lines = ring_table[(ring_table['comm'] == comm) & (ring_table['nranks'] == nranks)]['topo_line'].tolist()
    tree_lines = tree_table[(tree_table['comm'] == comm) & (tree_table['nranks'] == nranks)]['topo_line'].tolist()
    conn_lines = conn_table[(conn_table['comm'] == comm) & (conn_table['nranks'] == nranks)]['conn_line'].tolist()
    for line in (ring_lines + tree_lines + conn_lines):
        lines.append(line)
    return lines

def modify_label(in_txt, out_txt, rank_busId_mapping):
    fs = open(in_txt, 'r')
    lines = fs.readlines()
    fs.close()
    with open(out_txt, 'w') as f:
        for line in lines:
            if "fontsize" in line: 
                for m in re.finditer('"([^"]*)"', line):
                    rank_quot = line[m.start(0):m.end(0)]
                    rank = line[m.start(0)+1:m.end(0)-1] # find the rank with no quotation
                    print(rank_quot, rank)
                    line = line.replace(rank_quot,'\"'+rank_busId_mapping[rank]+'\"', 1)
                    f.write("{}".format(line))
                    break
            else:
                f.write("{}".format(line)) 


def coll_trace_processor(coll_trace_lines, group_list, coll_table):
    def process_string(line):
        split_list = line.split("[")
        return split_list[1].split("]")[0]
    # First pass
    results, func_name = {}, {}
    busIds = []
    RCCL_2_8 = False
    for line in coll_trace_lines:
        if ((" ## " in line) and ("KL HWID" in line or "KE" in line or "CE" in line)):  # RCCL From ROCm 4.3
            split_list = line.split(" ")
            t = float(process_string(split_list[5])) # seconds.microseconds
            rk, blk = process_string(split_list[6]).split(":") # rank:block_id
            op = int(split_list[7], 16) # hex to decimal
            ####### Kernel Launch #######
            if split_list[8] == 'KL':
                if "busId" in split_list:
                    busId = split_list[split_list.index("busId") + 1]
                    nranks = split_list[split_list.index("nRanks") + 1]
                    KL_key = str(op) + "," + str(busId) + "," + str(nranks) + ",t0"
#                     KL_key = str(op) + "," + str(busId) + ",t0"
                else:
                    # RCCL 2.8 or older
                    busId = int(rk)
                    RCCL_2_8 = True
                    KL_key = str(op) + "," + str(busId) + ",t0"
                if KL_key not in results or results[KL_key] > t:
                    results[KL_key] = t
                if busId not in busIds:
                     busIds.append(busId)
                if op not in func_name:
                     func_name[op] = split_list[11] # only work for KL
                        
            ####### Kernel End #######
            elif split_list[8] == 'KE':
                if "busId" in split_list:
                    busId = split_list[split_list.index("busId") + 1]
                    nranks = split_list[split_list.index("nRanks") + 1]
                    KE_key = str(op) + "," + str(busId) + "," + str(nranks) + ",t1"
#                     KE_key = str(op) + "," + str(busId) + ",t1"
                else:
                    # RCCL 2.8 or older
                    busId = int(rk)
                    RCCL_2_8 = True
                    KE_key = str(op) + "," + str(busId) + ",t1"
                if KE_key not in results or results[KE_key] < t:
                    results[KE_key] = t
                    
                    
            elif split_list[8] == 'CE':
                if "busId" in split_list:
                    busId = split_list[split_list.index("busId") + 1]
                    nranks = split_list[split_list.index("nRanks") + 1]
                    CE_key = str(op) + "," + str(busId) + "," + str(nranks) + ",t1"
#                     CE_key = str(op) + "," + str(busId) + "," + ",t1"
                else:
                    # RCCL 2.8 or older
                    busId = int(rk)
                    RCCL_2_8 = True
                    CE_key = str(op) + "," + str(busId) + ",t1"
                    
                if CE_key not in results or results[CE_key] < t:
                    results[CE_key] = t

        elif ((" ## " in line) and ("Abort" in line)):
            raise AssertionError("Abort")
    
    

    # Second pass
    bw_tables, time_tables = [], []
    for i, group in enumerate(group_list):
        group = list(group)
        # final_list = []
        bw_list = []
        time_list = []
        for op in func_name:
            found = False
            time_start, time_end = 1e9, 0
            if RCCL_2_8:
                for busId in range(len(group)):
                    key = str(op) + "," + str(busId)
                    if (key + ",t0") in results:
                        found = True
                        if busId == 0:
                            temp = [op, func_name[op]] # opCount, function name
                        t_start = results[key + ",t0"]
                        t_end = results[key + ",t1"]
                        temp.append(t_end - t_start) 
                        if t_start < time_start:
                            time_start = t_start
                        if t_end > time_end:
                            time_end = t_end
            else:
                for busId in group:
                    key = str(op) + "," + str(busId) + "," + str(len(group))
                    if (key + ",t0") in results:
                        found = True
                        if busId == group[0]:
                            temp = [op, func_name[op]] # opCount, function name
                        t_start = results[key + ",t0"]
                        t_end = results[key + ",t1"]
                        temp.append(t_end - t_start)
                        if t_start < time_start:
                            time_start = t_start
                        if t_end > time_end:
                            time_end = t_end
            if found:
                algobw_factor_times_size = coll_table[(coll_table['opCount'] == op) & (coll_table['nranks'] == len(group))]['algobw_factor_times_size'].unique()[0]
                data_size = coll_table[(coll_table['opCount'] == op) & (coll_table['nranks'] == len(group))]['data_size'].unique()[0]
                bw = temp
                bw[2:] = algobw_factor_times_size / bw[2:] /1e9
                bw = bw + [data_size, algobw_factor_times_size/(time_end - time_start)/1e9]
                temp.append(data_size)
                time_list.append(temp)
                bw_list.append(bw)
        
        time_tables.append(pd.DataFrame(time_list, columns = ['opCount', 'Function Name'] + group + ['data_size']))
        bw_tables.append(pd.DataFrame(bw_list, columns = ['opCount', 'Function Name'] + group + ['data_size', 'algBW']))
    return time_tables, bw_tables

def rccl_log_process():
    debug_log = os.path.abspath(args.rccl_debug_log)
    device_grouping_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "device_groups.txt")
    coll_table, conn_table, comm_table, ring_table, tree_table, coll_trace_lines = create_table(debug_log)
    device_group_list = device_grouping(comm_table, conn_table)
    with open(device_grouping_output, 'w') as f:
        for mySet in device_group_list:
            f.write("%s\n" % str(mySet))    
    
    for i, group in enumerate(device_group_list):
        group = list(group)
        print(group)
        temp_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp_{}.txt".format(i))
        temp1_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp_1_{}.txt".format(i))
        temp2_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp_2_{}.txt".format(i))
        nranks = len(group)
        lines = []
        rank_busId_mapping = {}
        for device in group:
            comms = comm_table[(comm_table['nranks'] == nranks) & (comm_table['busId'] == device)]['comm'].unique()
            assert len(comms) == 1       # if not 1, it means different devices may have same nRanks and busId.
            print(comms[0])
            lines = collect_topo(comms[0], nranks, ring_table, tree_table, conn_table, lines)
            rank = comm_table[(comm_table['nranks'] == nranks) & (comm_table['busId'] == device)]['rank'].unique()[0]
            busId = comm_table[(comm_table['nranks'] == nranks) & (comm_table['busId'] == device)]['busId'].unique()[0]
            rank_busId_mapping[rank] = busId
        
        with open(temp_txt, 'w') as f:
            for line in lines:
                f.write("{}\n".format(line))
                
        p1 = subprocess.Popen("awk -f extract_topo.awk {} > {}".format(temp_txt, temp1_txt), stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        time.sleep(1)
        modify_label(temp1_txt, temp2_txt, rank_busId_mapping)
        p2 = subprocess.Popen("dot -Tpng {} -o '{}.png'".format(temp2_txt, i), stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    
    ########### Collective trace processor ###########
    
    coll_table['algobw_factor_times_size'] = coll_table.apply(lambda row: algobw_factor_times_size(row['coll'], row['nranks'], row['data_size']), axis=1)
    coll_table = coll_table[['opCount','nranks','algobw_factor_times_size', 'data_size']].drop_duplicates(subset = ['opCount', 'data_size'])
    time_tables, bw_tables = coll_trace_processor(coll_trace_lines, device_group_list, coll_table)
    for i, (time_table, bw_table) in enumerate(zip(time_tables, bw_tables)):
        time_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "time_{}.csv".format(i))
        bw_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bw_{}.csv".format(i))
        time_table.to_csv(time_csv)
        bw_table.to_csv(bw_csv)
    


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rccl-debug-log", type=str, required=True, \
                            help="RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL RCCL_KERNEL_COLL_TRACE_ENABLE=1 executable")
    args = parser.parse_args()
    
    #### Requirement check #### 
    required = {'pandas'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
    ##### TODO #####
    # sudo apt-get install -y gawk
    # sudo apt-get install -y graphviz 
    import pandas as pd
    rccl_log_process()
    
# python final.py --rccl-debug-log gpt2_rccl_mp4_log_newPR.txt
# python extract_topo.py --rccl-debug-log base_2.9.log

# awk -f extract_topo.awk temp_0.txt > temp_1_0.txt
# awk -f extract_topo.awk 4_nodes.log.txt > test.txt