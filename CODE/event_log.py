import os
import shutil
import threading

import wmi  # Import the wmi library

from logicytics import Log, DEBUG

# Note: This script CANNOT be run without admin privileges

if __name__ == "__main__":
    log = Log({"log_level": DEBUG})


@log.function
def parse_event_logs(log_type: str, output_file: str):
    """
    Parses Windows event logs of a specified type and writes them to an output file using WMI.

    Args:
        log_type (str): The type of event log to parse (e.g., 'Security', 'Application').
        output_file (str): The file path where the parsed event logs will be written.

    Raises:
        Exception: If there is an error opening or reading the event log, or writing to the output file.
    """
    log.info(f"Parsing {log_type} events (Windows Events) and writing to {output_file}, this may take a while...")
    try:
        # Initialize WMI connection
        c = wmi.WMI()

        # Query based on log_type ('Security', 'Application', or 'System')
        query = f"SELECT * FROM Win32_NTLogEvent WHERE Logfile = '{log_type}'"
        log.debug(f"Executing WMI query: {query}")

        # Open the output file for writing
        with open(output_file, 'w') as f:
            events = c.query(query)
            f.write(f"Total records: {len(events)}\n\n")
            log.debug(f"Number of events retrieved: {len(events)}")
            for event in events:
                event_data = {
                    'Event Category': event.Category,
                    'Time Generated': event.TimeGenerated,
                    'Source Name': event.SourceName,
                    'Event ID': event.EventCode,
                    'Event Type': event.Type,
                    'Event Data': event.InsertionStrings
                }
                f.write(str(event_data) + '\n\n')

        log.info(f"{log_type} events (Windows Events) have been written to {output_file}")
    except wmi.x_wmi as err:
        log.error(f"Error opening or reading the event log: {err}")
    except Exception as err:
        log.error(f"Fatal issue: {err}")


if __name__ == "__main__":
    try:
        if os.path.exists('event_logs'):
            shutil.rmtree('event_logs')
        os.mkdir('event_logs')
    except Exception as e:
        log.error(f"Fatal issue: {e}")
        exit(1)

    threads = []
    threads_items = [('Security', 'event_logs/Security_events.txt'),
                     ('Application', 'event_logs/App_events.txt'),
                     ('System', 'event_logs/System_events.txt')]

    for log_type_main, output_file_main in threads_items:
        thread = threading.Thread(target=parse_event_logs, args=(log_type_main, output_file_main))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
