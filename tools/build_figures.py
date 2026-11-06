from pathlib import Path
import json,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parents[1]
A=R/'docs/assets'
if A.exists() and any(A.glob('*.png')):raise SystemExit('Preserve existing figures; rebuild only in a fresh candidate copy with empty assets.')
A.mkdir(parents=True,exist_ok=True)
d=json.loads((R/'evidence/results.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
colors=['#276B92','#DD793A']
for ax,role,title in zip(axes,['final','historical_best'],['Fixed step 200 (primary)','Original online-dev selection (secondary)']):
 s=d['statistics'][role];means=[s[v]['mean']*100 for v in 'UA'];sd=[s[v]['sample_sd']*100 for v in 'UA']
 ax.bar(['U: uniform','A: adaptive'],means,yerr=sd,capsize=5,color=colors,width=.55)
 for i,v in enumerate('UA'):
  values=[r['accuracy']*100 for r in d['results'] if r['role']==role and r['variant']==v]
  ax.scatter([i-.06,i,i+.06],values,c='black',s=14,zorder=5)
  ax.text(i,means[i]+sd[i]+.15,f'{means[i]:.2f}%',ha='center',fontsize=9)
 ax.axhline(1252/1319*100,color='#666',ls='--',lw=.8,label='Base 94.92%')
 ax.set_title(title,fontsize=10);ax.set_ylim(92.8,97);ax.set_ylabel('Strict accuracy (%)');ax.legend(loc='lower left',fontsize=8)
fig.suptitle('GSM8K test: 1,319 shared items; n=3 training seeds; sample SD',fontsize=11)
fig.text(.5,.02,'Points = individual seeds. Project numeric/format/EOS scoring; no significance claim.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(A/'test-results.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(9,3.2))
for ax,metric,title,unit in zip(axes,['gpu_hours','training_tokens','mixed_group_rate'],['Recorded training cost','Training response tokens','Mixed-reward groups'],['card-hours','million tokens','percent']):
 vals=[d['training_cost'][v][metric]['mean'] for v in 'UA']
 if metric=='training_tokens':vals=[x/1e6 for x in vals]
 if metric=='mixed_group_rate':vals=[x*100 for x in vals]
 ax.bar(['U','A'],vals,color=colors,width=.55);ax.set_title(title,fontsize=10);ax.set_ylabel(unit);ax.set_ylim(0,max(vals)*1.25)
 for i,x in enumerate(vals):ax.text(i,x+max(vals)*.04,f'{x:.2f}',ha='center')
fig.suptitle('Mean over three seeds; costs include recorded setup/resume/eval/save',fontsize=11)
fig.text(.5,.01,'More mixed groups did not establish a test or equal-budget efficiency advantage.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(A/'cost-and-signal.png');plt.close(fig)
rows=json.loads((R/'evidence/historical-online-dev.json').read_text(encoding='utf-8'))['rows']
fig,axes=plt.subplots(1,2,figsize=(9,3.2),sharey=True)
for ax,key,scale,label in zip(axes,['tokens','gpu_hours'],[1e6,1],['Cumulative training response tokens (M)','Recorded cumulative card-hours']):
 for variant,col in zip('UA',colors):
  for seed,style in zip([42,43,44],['-',':','--']):
   r=sorted([r for r in rows if r['variant']==variant and r['seed']==seed],key=lambda r:r[key])
   ax.plot([x[key]/scale for x in r],[x['accuracy']*100 for x in r],style,color=col,lw=1.2,alpha=.85,label=f'{variant}/{seed}')
 ax.set_xlabel(label);ax.set_ylabel('Historical online dev accuracy (%)');ax.legend(ncol=2,fontsize=7)
fig.suptitle('Exploratory historical online dev (512 items): noisy generation protocol',fontsize=11)
fig.text(.5,.005,'Separate per-run observations; no interpolated equal-budget claim. NOT reference-v4/test learning curves.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(A/'historical-budget-curves.png');plt.close(fig)
print('FIGURES_CREATED',len(list(A.glob('*.png'))))
