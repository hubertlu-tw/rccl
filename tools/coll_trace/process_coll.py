import os
import sys
import subprocess
import argparse
import pkg_resources

def main():
    def get_coll_trace_info(log_file):
        fs = open(log_file, 'r')
        lines = fs.readlines()
        fs.close()

        def process_string(line):
            split_list = line.split("[")
            return split_list[1].split("]")[0]

        # First pass
        results, func_name = {}, {}
        coll_trace_lines, busIds = [], []
        for j in range(len(lines)):
            line = lines[j].rstrip()
            #TODO: include "CE": collective end
    #         if ("KL hwid" in line or "KE" in line):  # RCCL Up to ROCm 4.2
            if ((" ## " in line) and ("KL HWID" in line or "KE" in line or "CE" in line)):  # RCCL From ROCm 4.3
                coll_trace_lines.append(line)
                split_list = line.split(" ")
                t = float(process_string(split_list[5])) # seconds.microseconds
                rk, blk = process_string(split_list[6]).split(":") # rank:block_id
                op = int(split_list[7], 16) # hex to decimal
                if split_list[8] == 'KL':
                    if len(split_list) >= 17 and split_list[16] == "busId":
                        busId = split_list[17]
                    elif len(split_list) >= 19 and split_list[18] == "busId":
                        busId = split_list[19]
                    else:
                        raise AssertionError("ERROR bad function index")
                    KL_key = str(op) + "," + str(busId) + ",t0"
                    if KL_key not in results or results[KL_key] > t:
                        results[KL_key] = t
                    if busId not in busIds:
                         busIds.append(busId)
                    if op not in func_name:
                         func_name[op] = split_list[11] # only work for KL
                elif split_list[8] == 'KE':
                    busId = split_list[10]
                    KE_key = str(op) + "," + str(busId) + ",t1"
                    if KE_key not in results or results[KE_key] < t:
                        results[KE_key] = t
                elif split_list[8] == 'CE':
                    if len(split_list) >= 15 and split_list[14] == "busId":
                        busId = split_list[15]
                    elif len(split_list) >= 17 and split_list[16] == "busId":
                        busId = split_list[17]
                    else:
                        raise AssertionError("ERROR bad function index")
                    busId = split_list[10]
                    KE_key = str(op) + "," + str(busId) + ",t1"
                    if KE_key not in results or results[KE_key] < t:
                        results[KE_key] = t
            elif ((" ## " in line) and ("Abort" in line)):
                raise AssertionError("Abort")

        # Second pass
        final_list = []
        for op in func_name:
            temp = [func_name[op]]
            for busId in busIds:
                t_start = results[str(op) + "," + str(busId) + ",t0"]
                t_end = results[str(op) + "," + str(busId) + ",t1"]
                temp.append(t_end - t_start)
            final_list.append(temp)

        return coll_trace_lines, func_name, busIds, results, pd.DataFrame(final_list, columns = ['Function Name'] + busIds) 

    def min_max_scaling(df, busIds):
        # copy the dataframe
        df_norm = df[busIds].copy()
        # apply min-max scaling
        for column in df_norm.columns:
            df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())        
        return df_norm

    
    debug_log = os.path.abspath(args.rccl_debug_log)
    required = {'pandas'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        
    import pandas as pd
    timing_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "timings.csv")
    max_time_device_ranking = os.path.join(os.path.dirname(os.path.realpath(__file__)), "max_time_device_ranking.png")
    
    coll_trace_lines, func_name, busIds, results, table = get_coll_trace_info(debug_log)
    table.index.name = "opCount"
    table.to_csv(timing_csv)
    
    # call the min_max_scaling function
    table_normalized = min_max_scaling(table, busIds)
    table_normalized['max_device_busId'] = table_normalized.idxmax(axis=1)
    ax = table_normalized['max_device_busId'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Device with max time spent in a kernel")
    _ = ax.set_xlabel('Device PCI Bus ID')
    _ = ax.set_ylabel('Frequency')
    
    fig = ax.get_figure()
    fig.savefig(max_time_device_ranking)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rccl-debug-log", type=str, required=True, \
                            help="RCCL log after running app with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL RCCL_KERNEL_COLL_TRACE_ENABLE=1 executable")
    args = parser.parse_args()
    main()
    
#     python process_coll.py --rccl-debug-log gpt2_rccl_mp4_log_coll_trace.txt