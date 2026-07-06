from datetime import datetime as dt
from pm4py.sim import play_out

def play_log(tree, config, scheduler_type = 'play_out'):
    if scheduler_type == 'play_out':
        log = play_out(tree, parameters={
            "num_traces": config["num_traces"
            ]})
    else:
        raise Exception("Unknown scheduler type")

    for i, trace in enumerate(log):
        trace.attributes['concept:name'] = str(i)
        for j, event in enumerate(trace):
            event['time:timestamp'] = dt.now()
            event['lifecycle:transition'] = "complete"
    return log