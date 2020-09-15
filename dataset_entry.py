class DsEntry:

    def __init__(self):
        self.user_id = ""
        self.text_id = -1
        self.input_values = []
        self.output_value = -1
        self.predicted_class = -1

    def __init__(self, user_id, text_id, entry_input, output):
        self.user_id = user_id
        self.text_id = text_id
        self.input_values = entry_input
        self.output_value = output
        self.predicted_class = -1


def get_entries_of(entries, expected_class):
    output = []

    for e in entries:
        e: DsEntry
        if e.output_value == expected_class:
            output.append(e)
    return output

def map_predicted_values(entries, precited_values):
    for i in range(len(entries)):
        entries[i].predicted_class = precited_values[i]
    return entries






