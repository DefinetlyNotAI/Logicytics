import os
import shutil
import threading

import wmi  # Import the wmi library

from logicytics import log


@log.function
def parse_event_logs(log_type: str, output_file: str):
    """
    Parses Windows event logs of a specified type and writes them to an output file using WMI.
    
    Args:
        log_type (str): The type of event log to parse (e.g., 'Security', 'Application', 'System').
        output_file (str): The file path where the parsed event logs will be written.
    
    Raises:
        wmi.x_wmi: If there is a WMI-specific error during event log retrieval.
        Exception: If there is a general error during file operations or log parsing.
    
    Notes:
        - Requires administrative privileges to access Windows event logs.
        - Retrieves all events for the specified log type using a WMI query.
        - Writes event details including category, timestamp, source, event ID, type, and data.
        - Logs informational and debug messages during the parsing process.
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
        thread.daemon = True  # Don't hang if main thread exits
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join(timeout=600)  # Wait max 10 minutes per thread
        if thread.is_alive():
            log.error(f"Thread for {thread.name} timed out (10 minutes)!")
