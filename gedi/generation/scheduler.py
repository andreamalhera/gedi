from datetime import datetime as dt, timedelta, timezone
from pm4py.sim import play_out

# Matches the default time:timestamp declared in the XES globals of the generator
START_TIMESTAMP = dt(1970, 1, 1, 1, 0, tzinfo=timezone(timedelta(hours=1)))

def play_log(tree, config, scheduler_type = 'play_out'):
    if scheduler_type == 'play_out':
        log = play_out(tree, parameters={
            "num_traces": config["num_traces"
            ]})
    else:
        raise Exception("Unknown scheduler type")

    # N * (i/N)^time_spread seconds: 1 (default) gives one second between events, values above 1
    # bunch events at the start of the log, values below 1 at the end.
    time_spread = config.get("time_spread", 1)
    if time_spread <= 0:
        raise ValueError(f"time_spread must be positive, got {time_spread}")
    total_events = sum(len(trace) for trace in log)
    n_events = 0
    previous_offset = -1
    for i, trace in enumerate(log):
        trace.attributes['concept:name'] = str(i)
        for event in trace:
            offset = round(total_events * 1e6 * (n_events / total_events) ** time_spread)
            offset = max(offset, previous_offset + 1) # At least one microsecond apart
            event['time:timestamp'] = START_TIMESTAMP + timedelta(microseconds=offset)
            event['lifecycle:transition'] = "complete"
            previous_offset = offset
            n_events += 1
    return log
