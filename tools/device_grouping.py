import os
import sys
import subprocess
import argparse
import pkg_resources

def main():
    def get_useful_info(log_file):
        fs = open(log_file, 'r')
        lines = fs.readlines()
        fs.close()
        conn_lines, comm_lines = [], []
        for j in range(len(lines)):
            line = lines[j].rstrip()
            if ("Channel" in line and "via" in line):
                conn_lines.append(line)
            if ("Init COMPLETE" in line and "busId" in line):
                comm_lines.append(line)
        return conn_lines, comm_lines


    ## It works for connection information like XXX via "P2P/direct pointer%s".(useReadStr),       --> pointer%s will not be collected
    ##                                                  "P2P/IPC%s".(useReadStr),  
    ##                                                  "P2P/indirect/%d[%lx]%s".(intermediateRank,comm->peerInfo[intermediateRank].busId, useReadStr) 
    ##                                                  "direct shared memory"                     --> shared memory
    ## However, it will not capture the information for CollNet for now.

    ## xxx [4] NCCL INFO Channel 00 : 0[e3000] -> 1[c3000] via P2P/IPC comm 0x7f53bc000e50 nRanks 04',  #(new output)
    def conn_table(conn_lines):
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
            nranks.append(split_list.index("nRanks") + 1)

        dict_conn = {'start_rank': start_rank, 'start_busid': start_busid, 'end_rank': end_rank, 'end_busid': end_busid, 'connection': connection, 'comm':comm, 'nranks':nranks}      
        return pd.DataFrame(dict_conn)

    def comm_table(comm_lines):
        comm, rank, nranks, cudaDev, busId = [], [], [], [], []   
        for line in comm_lines:
            split_list = line.rstrip().split(" ")
            comm.append(split_list[5])
            rank.append(split_list[7])
            nranks.append(split_list[9])
            cudaDev.append(split_list[11])
            busId.append(split_list[13])
        dict_comm = {'comm':comm, 'rank':rank, 'nranks':nranks, 'cudaDev':cudaDev, 'busId':busId}  
        return pd.DataFrame(dict_comm)


    def create_table(log_name):
        log_file = os.path.abspath(log_name)
        conn_lines, comm_lines = get_useful_info(log_file)
        return conn_table(conn_lines), comm_table(comm_lines) 
    
    ########
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

    def device_grouping(comm_table, conn_table):
        groups = []
        for index, row in comm_table.iterrows():
            temp = [row['busId'], list(conn_table[(conn_table['comm'] == row['comm']) & (conn_table['start_busid'] == row['busId'])]['end_busid'].unique())]
            groups.append(temp)
            # 1. We will use UnionFind to collect devices till the number of devices in a same group meets its nranks
            # 2. then we move         
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

    #### Requirement check #### 
    
    required = {'pandas'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        
    import pandas as pd
    
    
    
    debug_log = os.path.abspath(args.rccl_debug_log)
    device_grouping_output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "device_groups.txt")
#     print(debug_log)
    conn_table, comm_table = create_table(debug_log)
    device_group_list = device_grouping(comm_table, conn_table)
    with open(device_grouping_output, 'w') as f:
        for mySet in device_group_list:
            f.write("%s\n" % str(mySet))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rccl-debug-log", type=str, required=True, \
                            help="RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL RCCL_KERNEL_COLL_TRACE_ENABLE=1 executable")
    args = parser.parse_args()
    main()
    
#     python device_grouping.py --rccl-debug-log gpt2_rccl_mp4_log_pr.txt