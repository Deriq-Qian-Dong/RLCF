from rich.console import Console
from rich.table import Table
import numpy as np
import pandas as pd
import sys

compare_from=int(sys.argv[1])
compare_to=int(sys.argv[2])
tmp1=pd.read_csv("output/trlx/Psg_GeneratedQry_Reward-%d.tsv"%compare_from,sep='\t')
tmp2=pd.read_csv("output/trlx/Psg_GeneratedQry_Reward-%d.tsv"%compare_to, sep='\t')

reward2=np.array(tmp2['reward'])
reward1=np.array(tmp1['reward'])
reward2=reward2.reshape(-1,4)
reward1=reward1.reshape(-1,4)

mask = ((reward2-reward1)>=0).all(1)

prompt1 = np.array(tmp1['prompt']).reshape(-1,4)[mask]
prompt2 = np.array(tmp2['prompt']).reshape(-1,4)[mask]
output1 = np.array(tmp1['output']).reshape(-1,4)[mask]
output2 = np.array(tmp2['output']).reshape(-1,4)[mask]

reward2_=reward2[mask]
reward1_=reward1[mask]
reward2 = reward2_
reward1 = reward1_

print((reward2-reward1).sum(1).max())
# idx = np.argmax((reward2-reward1).sum(1))
idxs=np.argsort((reward2-reward1).sum(1))
idxs = idxs[::-1]

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("passages", width=50)
table.add_column("Summary before RL", max_width=100)
table.add_column("MRR",  max_width=5)
table.add_column("Summary after RL", max_width=100)
table.add_column("MRR",  max_width=5)
for idx in idxs[:10]:
    for psg,qry1,rd1,qry2,rd2 in zip(prompt1[idx], output1[idx], reward1[idx], output2[idx],reward2[idx]):
        table.add_row(psg, qry1, str(rd1), qry2, str(rd2), end_section=True)
    table.add_row("-",'-','-','-','-', end_section=True)

console.print(table)
