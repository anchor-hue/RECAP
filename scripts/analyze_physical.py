"""Fraction-only ProteoMixture analysis; reads saved proportions and fixed truth."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'data' / 'proteomixture'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result-dir',type=Path,default=ROOT / 'results' / 'physical')
parser.add_argument('--prep-dir',type=Path,default=PREP)
parser.add_argument('--out-dir',type=Path)
parser.add_argument('--seed',type=int,default=2021)
args = parser.parse_args()
root = args.result_dir.resolve()
out = args.out_dir or root/'analysis_output_fraction_only'
out.mkdir(parents=True,exist_ok=True)
cells = ['Lymphocyte','Stroma','Tumor']
designs = ['replicate_holdout_A','replicate_holdout_B','all_pure_reference']
labels = dict(zip(designs,['Reference A','Reference B','All-pure reference']))

def metrics(x,y):
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    return {'Pearson':float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else np.nan,
            'CCC':float(2*np.mean((x-x.mean())*(y-y.mean()))/(x.var()+y.var()+(x.mean()-y.mean())**2)) if x.std()>0 and y.std()>0 else np.nan,
            'RMSE':float(np.sqrt(np.mean((x-y)**2))), 'MAE':float(np.mean(np.abs(x-y))), 'NEntries':len(x)}

rows=[];mapping=[]
for design in designs:
    truth_path=args.prep_dir/design/'target_fractions.csv'
    run_dir=root/design/f'seed_{args.seed}'
    pred_path=run_dir/'target_celltype_proportions.csv'
    truth=pd.read_csv(truth_path,index_col=0).loc[:,cells]
    pred=pd.read_csv(pred_path,index_col=0).loc[truth.index,cells]
    order=pd.read_csv(args.prep_dir/design/'target_order.csv').set_index('ID')
    assert truth.index.is_unique and pred.index.is_unique
    assert np.isfinite(truth.to_numpy()).all() and np.isfinite(pred.to_numpy()).all()
    for target in truth.index:
        present=truth.loc[target]>0
        assert present.sum()==2
        series='Tumor-Stroma' if present.Stroma else 'Tumor-Lymphocyte'
        for cell in cells:
            x,y=float(truth.loc[target,cell]),float(pred.loc[target,cell])
            rows.append({'Design':design,'Target':target,'OriginalSample':order.loc[target,'OriginalSample'],
                         'MixtureSeries':series,'CellType':cell,'TrueFraction':x,'PredictedFraction':y,
                         'AbsoluteError':abs(x-y),'SquaredError':(x-y)**2})
    mapping.append({'Design':design,'ReferenceLabel':labels[design],'PredictedFractions':str(pred_path),
                    'PredefinedFractions':str(truth_path),'TargetOrder':str(args.prep_dir/design/'target_order.csv'),
                    'Config':str(run_dir/'config.json')})
long=pd.DataFrame(rows)
long.to_csv(out/'fraction_predictions_long.csv',index=False)
wide=long.pivot(index=['Design','Target','MixtureSeries'],columns='CellType',values=['TrueFraction','PredictedFraction'])
wide.columns=[f'{"True" if a=="TrueFraction" else "Pred"}_{b}' for a,b in wide.columns]
wide.reset_index().to_csv(out/'fraction_predictions_wide.csv',index=False)
summary=[];by_series=[];by_cell=[]
for design,g in long.groupby('Design',sort=False):
    m=metrics(g.TrueFraction,g.PredictedFraction)
    summary.append({'Design':design,'ReferenceLabel':labels[design],'FractionCorr':m['Pearson'],
                    'FractionCCC':m['CCC'],'FractionRMSE':m['RMSE'],'FractionMAE':m['MAE'],'NEntries':m['NEntries']})
    for series,t in g.groupby('MixtureSeries'):
        by_series.append({'Design':design,'MixtureSeries':series,**metrics(t.TrueFraction,t.PredictedFraction)})
        for cell,h in t.groupby('CellType'):
            by_cell.append({'Design':design,'MixtureSeries':series,'CellType':cell,**metrics(h.TrueFraction,h.PredictedFraction)})
pd.DataFrame(summary).to_csv(out/'proteomixture_overall_summary.csv',index=False)
pd.DataFrame(by_series).to_csv(out/'fraction_series_summary.csv',index=False)
pd.DataFrame(by_cell).to_csv(out/'fraction_series_component_summary.csv',index=False)
pd.DataFrame(mapping).to_csv(out/'file_mapping.csv',index=False)
(out/'analysis_scope.json').write_text(json.dumps({'endpoint':'component-fraction estimation','designs':designs,
 'scale':'predefined component proportions','ccc_moment_ddof':0,'evaluates_expression':False,
 'evaluates_reconstruction':False,'reruns_training':False,'n_entries_per_design':30},indent=2)+'\n')
print(pd.DataFrame(summary).to_string(index=False))
print('Saved fraction-only results:',out)
