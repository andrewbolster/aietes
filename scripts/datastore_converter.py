__author__ = 'bolster'
import bounos.Analyses.Trust
import aietes.Tools as Tools
import click
from collections import OrderedDict
import pandas as pd



def comms_log_to_trust(comms_log):
    trust_log = {
        k:
        bounos.Analyses.Trust.generate_trust_logs_from_comms_logs(v)
        for k,v in comms_log
    }
    return trust_log

@click.command()
@click.option('--comms_pkl', help="Comms Log to swap for hd5")
def comms_log_split_to_hdf5(comms_pkl):
    comms_log = Tools.unCpickle(comms_pkl)
    click.echo( "Loaded")
    inverted_logs = {}
    for var,runs in comms_log.iteritems():
        for run,nodes in runs.iteritems():
            for node, inner_logs in nodes.iteritems():
                for k,v in inner_logs.iteritems():
                    if not inverted_logs.has_key(k):
                        inverted_logs[k]={}
                    if not inverted_logs[k].has_key(var):
                        inverted_logs[k][var]={}
                    if not inverted_logs[k][var].has_key(run):
                        inverted_logs[k][var][run]={}
                    inverted_logs[k][var][run][node]=v
    dfs = {}
    del comms_log
    click.echo( "First Cycle")
    for k,v in inverted_logs.iteritems():
        v=OrderedDict(sorted(v.iteritems(), key=lambda _k:_k))
        try:

            if k not in []:
                # Var/Run/Node/(N/t) MultiIndex
                dfs[k]=pd.concat([
                    pd.concat([
                        pd.concat([pd.DataFrame(iiiv)
                                   for iiik,iiiv in iiv.iteritems()],
                                   keys=[iiik for iiik,iiiv in iiv.iteritems()])
                        for iik,iiv in iv.iteritems()],
                        keys=[iik for iik,iiv in iv.iteritems()]
                    )
                    for ik,iv in v.iteritems()],
                    keys=[ik for ik,iv in v.iteritems()],
                    names=['var','run','node','n']
                )
            else:
                # Var/Run/Node MultiIndex
                dfs[k]=pd.concat([
                    pd.concat([pd.DataFrame.from_dict(iiv, orient='index')
                               for iik,iiv in iv.iteritems()],
                               keys=[iik for iik,iiv in iv.iteritems()]
                    )
                    for ik,iv in v.iteritems()],
                    keys=[ik for ik,iv in v.iteritems()],
                    names=['var','run','node']
                )
        except:
            click.echo( "{k} didn't work".format(k=k))
            raise
    del inverted_logs
    click.echo( "Dumping to log_panel.h5")
    log_panel = pd.Panel.from_dict(dfs)
    log_panel.to_hdf('log_panel.h5','panel')

if __name__ == '__main__':
    comms_log_split_to_hdf5()