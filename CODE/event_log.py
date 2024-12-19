import os
from os import mkdir

import win32evtlog  # TODO - Remove this dependency, find a alternative

from logicytics import Log, DEBUG

# Note: This script CANNOT be run without admin privileges

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


@log.function
def parse_event_logs(log_type: str, output_file: str, server: str = 'localhost'):
    """
    Parses Windows event logs of a specified type and writes them to an output file.

    Args:
        log_type (str): The type of event log to parse (e.g., 'Security', 'Application').
        output_file (str): The file path where the parsed event logs will be written.
        server (str): The name of the server to connect to. Default is 'localhost'.

    Raises:
        Exception: If there is an error opening or reading the event log, or writing to the output file.
    """
    try:
        hand = win32evtlog.OpenEventLog(server, log_type)
        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        total = win32evtlog.GetNumberOfEventLogRecords(hand)

        with open(output_file, 'w') as f:
            f.write(f"Total records: {total}\n\n")
            events = win32evtlog.ReadEventLog(hand, flags, 0)
            while events:
                for event in events:
                    event_data = {
                        'Event Category': event.EventCategory,
                        'Time Generated': event.TimeGenerated.Format(),
                        'Source Name': event.SourceName,
                        'Event ID': event.EventID,
                        'Event Type': event.EventType,
                        'Event Data': event.StringInserts
                    }
                    f.write(str(event_data) + '\n\n')
                events = win32evtlog.ReadEventLog(hand, flags, 0)

        win32evtlog.CloseEventLog(hand)
        log.info(f"{log_type} events (Windows Events) have been written to {output_file}")
    except Exception as e:
        log.error(f"Fatal issue: {e}")


if __name__ == "__main__":
    if os.path.exists('event_logs'):
        os.rmdir('event_logs')
    mkdir('event_logs')
    parse_event_logs('Security', 'event_logs/Security_events.txt')
    parse_event_logs('Application', 'event_logs/App_events.txt')
    parse_event_logs('System', 'event_logs/System_events.txt')
